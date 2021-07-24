import './App.css';
import FederatedModel from './model/model';

function App() {
  const model = new FederatedModel();
  console.log(model.model);

  return (
    <div className="button-container">
      <button>Get and set new weights</button>
      <button>Send updated weights</button>
      <button>Train local model</button>
    </div>
  );
}

export default App;
