import './App.css';
import React, { useState } from 'react';
import axios from 'axios';
import * as tf from '@tensorflow/tfjs';
import FederatedModel from './model/model';
import { BE_LOCAL_URL, BE_URL } from './constants/constants';

function App() {
  const TOTAL_CLIENTS = 6;
  const [clientNum, setClientNum] = useState(0);
  const federatedModel = new FederatedModel();;

  const getNewWeights = async () => {
    const res = await axios({
      method: 'GET',
      url: `${BE_URL}/get-weights`
    });

    const json = JSON.parse(res.data.model);
    const weightData = new Uint8Array(Buffer.from(json.weightData, "base64")).buffer;
    const updatedModel = await tf.loadLayersModel(tf.io.fromMemory(json.modelTopology, json.weightSpecs, weightData));
    federatedModel.model.setWeights(updatedModel.getWeights());
    console.log('Done updating weights');
  }

  return (
    <div className="button-container">
      <button onClick={() => setClientNum((clientNum + 1) % TOTAL_CLIENTS)}>Current client num: {clientNum}</button>
      <button onClick={() => getNewWeights()}>Get and set new weights</button>
      <button onClick={() => federatedModel.train(clientNum)}>Train and send updated weights</button>
      <button onClick={() => federatedModel.testAccuracy()}>Test model accuracy</button>
    </div>
  );
}

export default App;
