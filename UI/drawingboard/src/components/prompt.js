import React from 'react'

const Prompt = () => {

  const library = ['book', 'car', 'bird', 'dog', 'flower', 'bottle']

  var currPrompt = library[Math.floor(Math.random() * library.length)]

  return (
    <div>
      <p>{currPrompt}</p>
    </div>
  );
}

export default Prompt