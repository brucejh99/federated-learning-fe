import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import axios from 'axios';
import { BE_LOCAL_URL, BE_URL } from '../constants/constants';
import { MnistData } from './mnistdata';

export default class FederatedModel {
    constructor() {
        this.model = this.modelBuilder();
        this.mnistdata = undefined;
    }

    // given input set of weights that fit in the model, load the weights into this.model
    setWeights(weights) {
        this.model.setWeights(weights);
    }

    // TODO: try creating hdf5 file and sending
    // return weights of the model in some standard format
    getWeights() {
        return this.model.getWeights();
    }

    // given some set of data (ex. image, label pair), perform federated learning prediction to
    // generate updated weights
    async train(clientNum) {
        console.log('Training...');
        if (this.mnistdata === undefined) {
            this.mnistdata = new MnistData(clientNum);
            const success = await this.mnistdata.load(clientNum); // is this how asyncs work in js?
        }

        const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
        const container = {
            name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
        };
        const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
        
        const BATCH_SIZE = 32; // hyperparameter?
        const TRAIN_DATA_SIZE = 2000;
        const TEST_DATA_SIZE = 5000;

        const [trainXs, trainYs] = tf.tidy(() => {
            const d = this.mnistdata.nextTrainBatch(TRAIN_DATA_SIZE);
            return [
                d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
                d.labels
            ];
        });

        const info = await this.model.fit(trainXs, trainYs, {
            batchSize: BATCH_SIZE,
            epochs: 5,
            shuffle: true,
            callbacks: fitCallbacks
        });
        console.log('Updated accuracy', info.history.acc);

        // https://stackoverflow.com/questions/55532746/tensorflow-nodejs-serialize-deserialize-a-model-without-writing-it-to-a-uri
        let result = await this.model.save(tf.io.withSaveHandler(async modelArtifacts => modelArtifacts));
        result.weightData = Buffer.from(result.weightData).toString("base64");
        const jsonStr = JSON.stringify(result);

        console.log('Sending updated weights...');
        const res = await axios({
            method: 'POST',
            url: `${BE_LOCAL_URL}/aggregate-weights`,
            data: {
                model: jsonStr,
                client: clientNum
            }
        });
        console.log('res', res);
    }

    // pre-designed model from: https://codelabs.developers.google.com/codelabs/tfjs-training-classfication/index.html#4
    // TODO: see if we need to update IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, etc. Keep the citation above.
    modelBuilder() {
        const model = tf.sequential();
        
        const IMAGE_WIDTH = 28;
        const IMAGE_HEIGHT = 28;
        const IMAGE_CHANNELS = 1;  
        
        // In the first layer of our convolutional neural network we have 
        // to specify the input shape. Then we specify some parameters for 
        // the convolution operation that takes place in this layer.
        model.add(tf.layers.conv2d({
            inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
            kernelSize: 5,
            filters: 8,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));
    
        // The MaxPooling layer acts as a sort of downsampling using max values
        // in a region instead of averaging.  
        model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
        
        // Repeat another conv2d + maxPooling stack. 
        // Note that we have more filters in the convolution.
        model.add(tf.layers.conv2d({
            kernelSize: 5,
            filters: 16,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));
        model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
        
        // Now we flatten the output from the 2D filters into a 1D vector to prepare
        // it for input into our last layer. This is common practice when feeding
        // higher dimensional data to a final classification output layer.
        model.add(tf.layers.flatten());
    
        // Our last layer is a dense layer which has 10 output units, one for each
        // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
        const NUM_OUTPUT_CLASSES = 10;
        model.add(tf.layers.dense({
            units: NUM_OUTPUT_CLASSES,
            kernelInitializer: 'varianceScaling',
            activation: 'softmax'
        }));
    
        
        // Choose an optimizer, loss function and accuracy metric,
        // then compile and return the model
        const optimizer = tf.train.adam();
        model.compile({
            optimizer: optimizer,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy'],
        });
    
        return model;
    }
}
