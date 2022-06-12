import logo from "./logo.svg";
import "./App.css";
import { useState } from "react";
import axios from "axios";

function App() {
  const [text, setText] = useState("");
  const [data, setData] = useState(null);
  const [dataList, setDataList] = useState([]);

  const [isLoading, setIsLoading] = useState(false);

  const analyseSentence = () => {
    setIsLoading(true);
    axios
      .post("/predict", {
        sentences: [text],
      })
      .then((res) => {
        setDataList([res.data, ...dataList]);
        setIsLoading(false);
      });
  };

  return (
    <div className="App">
      <div className="w-full align-center justify-center mt-2 text-center">
        <form className="bg-white shadow-md max-w-xl mx-auto rounded px-8 pt-6 pb-8 mb-4">
          <div className="mb-4">
            <label
              className="block text-gray-700 text-sm font-bold mb-2"
              for="username"
            >
              Sentence to be analysed
            </label>
            <input
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              id="sentence"
              type="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="I liked the food"
            />
          </div>
          <div className="flex items-center justify-center">
            <button
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline flex align-middle"
              type="button"
              disabled={isLoading}
              onClick={analyseSentence}
            >
              {isLoading ? (
                "Analysing..."
              ) : (
                "Analyse"
              )}
            </button>
          </div>
        </form>
        {dataList.map((data, index) => {
          return (
            <div className="bg-white shadow-md max-w-7xl mx-auto rounded px-8 pt-6 pb-8 mb-4 flex flex-wrap align-baseline">
              {data["tokens"].map((token, index) => {
                let classData = "mx-1 py-2";

                if (data["ate_outputs"][index] !== 0) {
                  classData = "mx-1 underline underline-offset 1 font-bold";
                }

                if (data["sa_outputs"][index] == 1) {
                  classData += " p-2 rounded-xl bg-red-500 font-bold";
                }

                if (data["sa_outputs"][index] == 2) {
                  classData += " p-2 rounded-xl bg-yellow-500 font-bold";
                }

                if (data["sa_outputs"][index] == 3) {
                  classData += " p-2 rounded-xl bg-green-500 font-bold";
                }

                return (
                  <span key={`token-${index}`} className={classData}>
                    {token}
                  </span>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default App;
