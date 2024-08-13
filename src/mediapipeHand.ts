// @ts-ignore Import module
import { HandLandmarker, FilesetResolver, DrawingUtils } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0';
//import * as ort from 'onnxruntime-web';

type Landmark = {
    x: number;
    y: number;
    z: number;
};

const ort = window.ort;

//load elements from html here
const video = document.getElementById("videoElement") as HTMLVideoElement;
const canvasElt = document.getElementById("canvasFrame") as HTMLCanvasElement;
const canvasContext = canvasElt.getContext("2d") as CanvasRenderingContext2D;
const enableButton = document.getElementById("enableBtn") as HTMLButtonElement;
const drawing_utils = new DrawingUtils(canvasContext);
const textBox = document.getElementById("textBox") as HTMLTextAreaElement;

enableButton.addEventListener("click", startVideo);

async function startVideo(){
    try {
        navigator.mediaDevices.getUserMedia({video:true, audio:false})
        .then(function(stream){
            video.srcObject = stream;
            video.play();
        })
    }
    catch (err) {
        console.log("Permission denied" + err);
    }
}

//define variables needed for mediapipe implementation
let lastTime = -1;
let results: any;
let handLandmarker: any = null;

// loading model
let session: ort.InferenceSession;

async function loadModel() {
    try {
        if (typeof ort === 'undefined') {
            throw new Error('ONNX Runtime Web is not loaded');
        }
        
        // Use ort.InferenceSession.create
        session = await ort.InferenceSession.create('./sign_language_model.onnx');
        console.log('Model loaded successfully');
    } catch (e) {
        console.error("Failed to load ONNX model:", e);
    }
}
window.addEventListener('load', loadModel);

// wait for HandLandmarker class to finish loading
async function awaitHandLandmarker() {
    try {
        const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm");
        handLandmarker = await HandLandmarker.createFromOptions(vision, { baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", 
            delegate: "GPU"        
            },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
        });
    } catch (error) {
        console.error("Error", error);
    }
}

const waitForHandLandmarker = (callback:any) => {
    const interval = setInterval(() => {
        if (handLandmarker !== null) {
            clearInterval(interval);
            callback();
        }
    }, 100); // Check every 100ms
};

awaitHandLandmarker().then(() => {
    waitForHandLandmarker(() => {
        drawWebcam();
    });
});

function drawWebcam(){
    //set canvas dimensions to video dimensions
    canvasElt.width = video.videoWidth;
    canvasElt.height = video.videoHeight;

    //set current time to last time and results to processing for the current frame of the video
    let startTime = performance.now();
    if (lastTime != video.currentTime){
        lastTime = video.currentTime;
        results = handLandmarker.detectForVideo(video, startTime);
    }

    canvasContext.clearRect(0, 0, canvasElt.width, canvasElt.height);

    //puts video onto canvas
    canvasContext.save();
    canvasContext.scale(-1,1);
    canvasContext.translate(-canvasElt.width, 0);
    canvasContext.drawImage(video, 0, 0, canvasElt.width, canvasElt.height);
    canvasContext.restore();

    if (results.landmarks && results.landmarks.length > 0) {
        for (const landmarks of results.landmarks){
            const mirroredLandmarks = landmarks.map((landmark: any) => ({
                ...landmark,
                x: 1 - landmark.x
            }));
            drawing_utils.drawConnectors(mirroredLandmarks, HandLandmarker.HAND_CONNECTIONS, {color: '#62C6F2', lineWidth: 3});
            drawing_utils.drawLandmarks(mirroredLandmarks, {color: "white", lineWidth: 1});
            canvasContext.restore();
            recognizeGesture(mirroredLandmarks);
        }
    }
    else {
        textBox.textContent = "No hand detected";
    }
    requestAnimationFrame(drawWebcam)
}

async function recognizeGesture(landmarks: Landmark[]){
    if (!session){
        console.log("Model not loaded")
        return
    }

    const input = new Float32Array(42);
    landmarks.forEach((landmark, index) => {
        input[index * 2] = landmark.x;
        input[index * 2 + 1] = landmark.y;
    });

    const tensorInput = new ort.Tensor('float32', input, [1, 42]);
    const output = await session.run({ input: tensorInput });
    const predictions = output.output.data as Float32Array;
    const maxIndex = predictions.indexOf(Math.max(...predictions));

    const letter = String.fromCharCode(65 + maxIndex);

    textBox.textContent = `Recognized Letter: ${letter}`;
}  

