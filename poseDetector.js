const initClassifier = () => {
  return knnClassifier.create();
};

const initMobileNet = async () => {
  return await mobilenet.load();
};

let classifier, net;

const run = () => {
  window.onload = () => {
    console.log("onload");
    classifier = initClassifier();
    net = initMobileNet();
    mobilenet.load().then((model) => {
      console.log("mobilenet loaded");
      console.log(model);
      // const image = document.getElementById('tennis');
      // console.log(model.infer(image, 'conv_preds'));
      knnClassifyWebcam(model);
      // classifyWebcam(model);
    });
  };
  // classifyWithMobileNet();
};

const classifyWithPoseNet = async () => {
  const image = document.getElementById('tennis');
  const pose = await net.estimateSinglePose(image);
  console.log(pose);
};

const classifyWithMobileNet = async () => {
  // classifyImage(net);
  // classifyWebcam(net);
};

const classifyImage = async (net) => {
  const image = document.getElementById('img');
  const result = await net.classify(image);
  console.log(result[0].className);
};

const classifyWebcam = async (net) => {
  const webcamElement = document.getElementById('webcam');
  const webcam = await tf.data.webcam(webcamElement);

  while (true) {
    const img = await webcam.capture();
    const result = await net.classify(img);

    console.log(result[0].className);
    console.log(result[0].probability);
    document.getElementById('console').innerText = `
      prediction: ${result[0].className}\n
      probability: ${result[0].probability}
    `;
    img.dispose();
    await tf.nextFrame();
  }
};

const knnClassifyWebcam = async (model) => {
  const webcamElement = document.getElementById('webcam');
  const webcam = await tf.data.webcam(webcamElement);
  console.log("knn func, net : ");

  document.getElementById('class-a').addEventListener('click', () => addExample(0, model));
  document.getElementById('class-b').addEventListener('click', () => addExample(1, model));
  document.getElementById('class-c').addEventListener('click', () => addExample(2, model));

  while (true) {
    if (classifier.getNumClasses() > 0) {
      webcam.capture().then(async (img) => {
        const activation = model.infer(img, 'conv_preds');
        const result = await classifier.predictClass(activation);
        console.log(result);

        const classes = ['A', 'B', 'C'];
        document.getElementById('console').innerText = `
        prediction: ${classes[result.label]}\n
        probability: ${result.confidences[result.label]}
      `;
        img.dispose();
      });
    }

    await tf.nextFrame();
  }
};

const addExample = async (classId, model) => {
  console.log("num classes");
  console.log(classifier.getNumClasses());
  const webcamElement = await document.getElementById('webcam');
  console.log(webcamElement);
  const webcam = await tf.data.webcam(webcamElement);
  console.log(webcam);
  let img = await webcam.capture();
  console.log(img);
  // const img = await webcam.capture().then(() => {
  //   // const activation = model.infer(img, 'conv_preds');
  //   // classifier.addExample(activation, classId);
  // }).catch((error) => console.log(error));
  // const webcamElement = document.getElementById('webcam');
  //
  //
  // img.dispose();
};

run();
