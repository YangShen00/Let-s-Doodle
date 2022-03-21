import React, { useRef, useState, useEffect, createRef, createFileName } from 'react';
import { useScreenshot } from 'use-react-screenshot'

import './App.css';
import Timer from './components/timer'
import Prompt from './components/prompt'

function App() {

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

  const minutes = 0
  const seconds = 5

  const [[mins, secs], setTime] = React.useState([minutes, seconds]);
  const [prompt, setPrompt] = React.useState("start")

  const library = ['book', 'car', 'bird', 'dog', 'flower', 'bottle']

  const tick = () => {
    if (mins === 0 && secs === 0) {
      var result = evaluate();

      if (result === true) {
        reset()
        setPrompt(library[Math.floor(Math.random() * library.length)])
        clearCanvas()
      }
    } else if (secs === 0) {
      setTime([mins - 1, 59]);
    } else {
      setTime([mins, secs - 1]);
    }
  };

  // here is the function to evaluate the result
  const evaluate = () => true;

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
    context.lineWidth = 5
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
        <button onClick={downloadScreenshot}>Evaluate drawing</button>
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

export default App;
