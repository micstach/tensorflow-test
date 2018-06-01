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
  epochs: 10
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