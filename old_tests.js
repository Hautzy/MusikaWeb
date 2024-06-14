document.addEventListener('DOMContentLoaded', async () => {
    // Load the Graph Model
    async function loadGraphModel() {
        try {
            const model = await tf.loadGraphModel('./dec2_model_web/model.json');
            console.log('Graph model loaded successfully');
            return model;
        } catch (error) {
            console.error('Error loading graph model:', error);
        }
    }

    async function exploreModelProperties(model) {
        console.log('Input tensors:', model.inputs);
        console.log('Output tensors:', model.outputs);

        if (model.executor && model.executor.weightMap) {
            const weightMap = model.executor.weightMap;
            const weightKeys = Object.keys(weightMap);
            console.log('Number of weights:', weightKeys.length);
            weightKeys.forEach((key, index) => {
                const tensor = weightMap[key][0];
                console.log(`Weight ${index + 1}:`, key, tensor.shape);
            });
        } else {
            console.log('No weights found in the model.');
        }
    }

    // Perform Inference
    async function performInference(model) {
        // Create a dummy input tensor (replace with actual data as needed)
        const inputTensor = tf.zeros([1, 224, 224, 3]); // Example shape, adjust as necessary

        // Perform inference
        const outputTensor = model.predict(inputTensor);

        // Print the output tensor
        outputTensor.print();

        // Dispose tensors to release memory
        inputTensor.dispose();
        outputTensor.dispose();
    }

    // Print the Model JSON
    async function printModelJSON(model) {
        await model.save(tf.io.withSaveHandler(async (artifacts) => {
            console.log('Model JSON:', artifacts.modelTopology);
            return { success: true };
        }));
    }

    // Preprocess Data
    function preprocessData(data) {
        // Normalize data
        const normalizedData = data.div(tf.scalar(255));
        return normalizedData;
    }

    // Evaluate Model Performance
    async function evaluateModel(model, testData, testLabels) {
        // Perform inference
        const predictions = model.predict(testData);

        // Calculate accuracy or other metrics
        const accuracy = predictions.argMax(-1).equal(testLabels.argMax(-1)).mean().dataSync()[0];
        console.log('Accuracy:', accuracy);

        // Dispose tensors to release memory
        testData.dispose();
        testLabels.dispose();
        predictions.dispose();
    }

    // Main function to run all tasks
    async function main() {
        const model = await loadGraphModel();

        if (model) {
            console.log(model.predict(tf.randomNormal([2, 1, 4, 64])));
            await exploreModelProperties(model);
            await printModelJSON(model);
            /*
            await performInference(model);


            // Example usage of preprocessing data
            const dummyData = tf.zeros([1, 224, 224, 3]);
            const preprocessedData = preprocessData(dummyData);
            preprocessedData.print();
            dummyData.dispose();
            preprocessedData.dispose();

            // Example usage of evaluating the model (using dummy data)
            const testData = tf.zeros([10, 224, 224, 3]); // Example shape, adjust as necessary
            const testLabels = tf.zeros([10, 1]); // Example shape, adjust as necessary
            await evaluateModel(model, testData, testLabels);*/
        }
    }

    // Run the main function
    main();
});