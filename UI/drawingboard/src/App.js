import { useRef, useState, useEffect } from 'react';
import './App.css';
import Timer from './components/timer'
import Prompt from './components/prompt'

function App() {

  const minSecs = { minutes: 1, seconds: 40 }
  console.log("minSecs")

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
        <button className="button" onClick={clearCanvas}>Clear</button>
        <Timer MinSecs={minSecs} />
        <Prompt />
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
