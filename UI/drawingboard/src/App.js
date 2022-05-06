import React, { useRef, useState, useEffect, createRef, createFileName } from 'react';
import { useScreenshot } from 'use-react-screenshot'

import './App.css';
import Timer from './components/timer'
import Prompt from './components/prompt'

function App() {

  const categories = {
    0: 'The Eiffel Tower',
    1: 'The Great Wall of China',
    2: 'airplane',
    3: 'angel',
    4: 'animal migration',
    5: 'basket',
    6: 'bread',
    7: 'candle',
    8: 'cloud',
    9: 'diving board',
    10: 'dog',
    11: 'zigzag',
    12: 'tree',
    13: 'wine glass',
    14: 'firetruck',
    15: 'eye',
    16: 'flower',
    17: 'guitar',
    18: 'toilet',
    19: 'ice cream',
    20: 'table',
    21: 'duck',
    22: 'knee',
    23: 'elephant',
    24: 'tornado',
    25: 'hexagon',
    26: 'umbrella',
    27: 'triangle',
    28: 'hand',
    29: 'violin'
  }

  // screenshot
  const [image, takeScreenShot] = useScreenshot({
    type: "image/jpeg",
    quality: 1.0
  });

  const download = (image, { name = "img", extension = "jpg" } = {}) => {
    const a = document.createElement("a");
    console.log(image)
    a.href = image;
    a.download = "output.png";
    a.click();
  };

  const downloadScreenshot = () => takeScreenShot(canvasRef.current).then(download);



  // timer

  // TODO: decrease time in each round
  const minutes = 0
  const seconds = 15

  const [[mins, secs], setTime] = React.useState([minutes, seconds]);
  const [prompt, setPrompt] = React.useState("flower")
  const [score, setScore] = React.useState(0)
  const [stop, setStop] = React.useState(false)

  const library = Object.values(categories)

  const tick = () => {
    if (secs % 10 == 0) {
      var canvas = document.getElementById('canvas');
      setDataUrl(canvas.toDataURL())
      console.log(dataUrl)
      var result = evaluate();
      if (result === prompt) {
        reset()
        setPrompt(library[Math.floor(Math.random() * library.length)])
        setScore(score + 1)
        clearCanvas()
      }

    }

    if (mins === 0 && secs === 0) {
      // TODO: game should end here
      setStop(true)
    } else if (secs === 0) {
      setTime([mins - 1, 59]);
    } else {
      evaluate()
      setTime([mins, secs - 1]);
    }
  };

  // TODO: the function to evaluate the result

  const [dataUrl, setDataUrl] = useState("")

  const requestOptions = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataUrl })
  };

  async function evaluate() {
    takeScreenShot(canvasRef.current)
    const response = await fetch('http://localhost:9000/evaluate', requestOptions);
    const data = await response.json();
    console.log(data)
  }

  const reset = () => setTime([parseInt(minutes), parseInt(seconds)]);


  React.useEffect(() => {
    const timerId = setInterval(() => tick(), 1000);
    return () => clearInterval(timerId);
  });


  // canvas
  const canvasRef = useRef(null)
  const contextRef = useRef(null)

  const [isDrawing, setIsDrawing] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = window.innerWidth * 2;
    canvas.height = window.innerHeight * 2;
    canvas.style.width = `${window.innerWidth}px`;
    canvas.style.height = `${window.innerHeight}px`;

    const context = canvas.getContext('2d')
    context.scale(2, 2)
    context.lineCap = "round"
    context.strokeStyle = "black"
    context.lineWidth = 50
    contextRef.current = context;
  }, [])

  const startDrawing = ({ nativeEvent }) => {
    const { offsetX, offsetY } = nativeEvent;
    contextRef.current.beginPath()
    contextRef.current.moveTo(offsetX, offsetY)
    setIsDrawing(true)
  }

  const finishDrawing = () => {
    contextRef.current.closePath()
    setIsDrawing(false)
  }

  const draw = ({ nativeEvent }) => {
    if (!isDrawing) {
      console.log('nooo')
      return;
    }

    const { offsetX, offsetY } = nativeEvent;
    contextRef.current.lineTo(offsetX, offsetY)
    contextRef.current.stroke()

  }

  function clearCanvas() {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d')
    context.clearRect(0, 0, canvas.width, canvas.height);
  }

  if (stop) {
    return (
      <div className="stopbackground">
        <p className="stop"> stopped</p>
        <p className="stop"> your score is {score}</p>
        <p className="stop"> please refresh to restart</p>
      </div>
    )
  }
  else {

    return (

      <div>

        <div className="App">
          <header className="App-header">
            <p>
              Lets Doodle
            </p>

          </header>
        </div>

        <div className="info">

          {/* <Timer MinSecs={minSecs} /> */}
          {/* <Prompt /> */}
          <div className="border">
            <p className='timer'>{`${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`}</p>
            <a className='prompt'>{prompt}</a>
            <button className="button" onClick={clearCanvas}>Clear</button>
          </div>
          <p>Current Score: {score}</p>
          {/* <button onClick={downloadScreenshot}>Download</button> */}
          {/* <button onClick={() => evaluate()}>Evaluate</button> */}
        </div>

        <canvas id="canvas"
          onMouseDown={startDrawing}
          onMouseUp={finishDrawing}
          onMouseMove={draw}
          ref={canvasRef}
        ></canvas>


      </div>

    );
  }

}

export default App;
