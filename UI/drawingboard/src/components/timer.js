import React, { useState, useEffect } from 'react'

const Timer = ({ MinSecs }) => {

  const { minutes = 0, seconds = 60 } = MinSecs;
  const [[mins, secs], setTime] = React.useState([minutes, seconds]);
  const [prompt, setPrompt] = React.useState("start")

  const library = ['book', 'car', 'bird', 'dog', 'flower', 'bottle']

  const tick = () => {
    if (mins === 0 && secs === 0) {
      var result = evaluate();

      if (result == true) {
        reset()
        setPrompt(library[Math.floor(Math.random() * library.length)])
      }
    } else if (secs === 0) {
      setTime([mins - 1, 59]);
    } else {
      setTime([mins, secs - 1]);
    }
  };

  const evaluate = () => true;

  const reset = () => setTime([parseInt(minutes), parseInt(seconds)]);


  React.useEffect(() => {
    const timerId = setInterval(() => tick(), 1000);
    return () => clearInterval(timerId);
  });

  return (
    <div>
      <p>{`${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`}</p>
      <p>{prompt}</p>
    </div>
  );

}

export default Timer