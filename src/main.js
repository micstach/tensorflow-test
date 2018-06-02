import * as tf from '@tensorflow/tfjs';

// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
var x = [];
var y = [];

for (var i=0.0; i<10.0; i+=1.0) {
  x.push(1.0 * i);
  y.push(2.0 * x[i]);

  console.log(`f(${x[i]}) = ${y[i]}`);
}

const xs = tf.tensor1d(x);
const ys = tf.tensor1d(y);

// Train the model using the data.
model.fit(xs, ys, {
  epochs: 4
}).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  for (var i=0.0; i<=12.0; i+= 1.0) {
    const output = model.predict(tf.tensor2d([i], [1, 1]));
    var result = Array.from(output.dataSync())[0]; 
    console.log(`f(${i}) = ${result}`);
  }

  const output = model.predict(tf.tensor2d([4.5], [1, 1]));
  var result = Array.from(output.dataSync())[0]; 
  console.log(`f(${4.5}) = ${result}`);

});


// Define function
function predict(input) {
  // y = a * x ^ 2 + b * x + c
  // More on tf.tidy in the next section
  return tf.tidy(() => {
    const x = tf.scalar(input);

    const ax2 = a.mul(x.square());
    const bx = b.mul(x);
    const y = ax2.add(bx).add(c);

    return y;
  });
}

// Define constants: y = 2x^2 + 4x + 8
const a = tf.scalar(1);
const b = tf.scalar(-4);
const c = tf.scalar(4);

// Predict output for input of 2
const result = Array.from(predict(2).dataSync());
console.log(`result = ${result}`);
