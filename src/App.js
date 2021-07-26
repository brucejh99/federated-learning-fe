import './App.css';
import React, { useState } from 'react';
import axios from 'axios';
import * as tf from '@tensorflow/tfjs';
import FederatedModel from './model/model';
import { BE_LOCAL_URL } from './constants/constants';

function App() {
  const TOTAL_CLIENTS = 6;
  const [clientNum, setClientNum] = useState(0);

  const federatedModel = new FederatedModel();
  // console.log(federatedModel.model);

  const getNewWeights = async () => {
    const res = await axios({
      method: 'GET',
      url: `${BE_LOCAL_URL}/get-weights`
    });

    console.log(res);
    const json = JSON.parse(res.data.model);
    const weightData = new Uint8Array(Buffer.from(json.weightData, "base64")).buffer;
    const updatedModel = await tf.loadLayersModel(tf.io.fromMemory(json.modelTopology, json.weightSpecs, weightData));
    federatedModel.model.setWeights(updatedModel.getWeights());
    console.log('Done');
  }

  return (
    <div className="button-container">
      <button onClick={() => setClientNum((clientNum+1) % TOTAL_CLIENTS)}>Current client num: {clientNum}</button>
      <button onClick={() => getNewWeights()}>Get and set new weights</button>
      <button onClick={() => federatedModel.train(clientNum)}>Send updated weights</button>
      <button>Train local model</button>
    </div>
  );
}

export default App;
