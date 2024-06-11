const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');
 
async function predictClassification(model, image) {
    try {
        const tensor = tf.node
            .decodeJpeg(image)
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat()
 
            //const classes = ['Melanocytic nevus', 'Squamous cell carcinoma', 'Vascular lesion'];
            const classes = ['dark', 'mid-dark', 'mid-light', 'light'];
            // const classes = ['Beige', 'Cherry Blush', 'Cosmic Latte', 'Dark Grey', 'Dark pink', 'Dusty Pink',
            //                  'Grey', 'Lavender', 'Lavender blue', 'Light Sky blue', 'Mauve', 'Maroon', 'Navy', 'Pink peach', 'Purple',
            //                  'Red cherry', 'Red crimson', 'Rudy Pink', 'Sky blue', 'White']


 
        const prediction = model.predict(tensor);
        const score = await prediction.data();
        const confidenceScore = Math.max(...score) * 100;
 
        const classResult = tf.argMax(prediction, 1).dataSync()[0];
        const label = classes[classResult];
 
        let explanation 
        //,suggestion
        ;
 
        if(label === 'dark') {
            explanation = "ini adalah dark"
            //explanation = "Melanocytic nevus adalah kondisi di mana permukaan kulit memiliki bercak warna yang berasal dari sel-sel melanosit, yakni pembentukan warna kulit dan rambut."
            //suggestion = "Segera konsultasi dengan dokter terdekat jika ukuran semakin membesar dengan cepat, mudah luka atau berdarah."
        }
 
        if(label === 'mid-dark') {
            explanation = "ini adalah mid dark"
            //explanation = "Squamous cell carcinoma adalah jenis kanker kulit yang umum dijumpai. Penyakit ini sering tumbuh pada bagian-bagian tubuh yang sering terkena sinar UV."
            //suggestion = "Segera konsultasi dengan dokter terdekat untuk meminimalisasi penyebaran kanker."
        }
 
        if(label === 'mid-light') {
           explanation = "ini adalah mid-light"
            //explanation = "Vascular lesion adalah penyakit yang dikategorikan sebagai kanker atau tumor di mana penyakit ini sering muncul pada bagian kepala dan leher."
           //suggestion = "Segera konsultasi dengan dokter terdekat untuk mengetahui detail terkait tingkat bahaya penyakit."
        
        }
        if(label === 'light') {
            explanation = "light"
             //explanation = "Vascular lesion adalah penyakit yang dikategorikan sebagai kanker atau tumor di mana penyakit ini sering muncul pada bagian kepala dan leher."
            //suggestion = "Segera konsultasi dengan dokter terdekat untuk mengetahui detail terkait tingkat bahaya penyakit."
         
         }
 
        return { confidenceScore, label, explanation 
            //,suggestion 
        };
    } catch (error) {
        throw new InputError(`Terjadi kesalahan input: ${error.message}`)
    }
}
 
module.exports = predictClassification;