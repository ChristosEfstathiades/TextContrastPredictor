const whiteLuminance = 1.05
const blackLuminance = 0.05
let colourPicker;
let nn; 
function setup() {
    createCanvas(600, 400)
    colourPicker = createColorPicker('#000000')
    colourPicker.position(width/2, height/2)


    let storedNN = getItem('nn')
    if (storedNN !== null) {
        nn = NeuralNetwork.deserialize(storedNN)
    } else {
        nn = new NeuralNetwork(3, 3, 2)
        let RGBs = []
        for (let r = 0; r < 256; r+=1.6) {
            for (let g = 0; g < 256; g+=1.6) {
                for (let b = 0; b < 256; b+=1.6) {
                    RGBs.push([floor(r), floor(g), floor(b)])
                }
            }
        }
        // console.table(RGBs.splice(100000, 100100))
        // for (let iter = 0; iter < 5; iter++) {
        RGBs = shuffle(RGBs)
        for (let i = 0; i < RGBs.length; i++) {
            let r = RGBs[i][0]
            let g = RGBs[i][1]
            let b = RGBs[i][2]
            let normalisedInputs = [r / 255, g / 255, b / 255]
            let targetOutputs = contrastCheck(normalisedInputs)
            nn.train(normalisedInputs, targetOutputs)
            // console.log(r)
        }
        // }
        storeItem('nn', nn.serialize())
    }


}

function draw() {
    // frameRate(60)
    // noLoop()
    // testNN()
    background(colourPicker.color())

    rgbCode = colourPicker.color().levels.splice(0,3).map(function(colour) {
        return colour / 255
    })
    result = nn.feedForward(rgbCode)
    // lol = contrastCheck(rgbCode)
    // console.log(result)
    if (result[0] > result[1]) {
        fill(255, 255, 255)
    } else {
        fill(0, 0, 0)
    } 

    circle(width/2-50, height-50, 50)
    
}

function testNN() {
    let correct = 0
    let incorrect = 0
    for (let i = 0; i < 1000000; i++) {
        let rgbNormalised = [floor(random(0, 255))/255, floor(random(0, 255))/255, floor(random(0, 255))/255]
        let targetOutput = contrastCheck(rgbNormalised)
        let nnOutput = nn.feedForward(rgbNormalised)

        if (nnOutput[0] > nnOutput[1]) {
            nnOutput = [1, 0]
        } else {
            nnOutput = [0, 1]
        }

        if (targetOutput[0] == nnOutput[0] && targetOutput[1] == nnOutput[1]) {
            correct += 1
        } else {
            incorrect += 1
        }

        console.log((100 / (correct + incorrect)) * correct)
    }
}

function contrastCheck(rgb) {
    // let RGBluminance = luminance(rgb)

    // whiteContrast = whiteLuminance / RGBluminance
    // blackContrast = RGBluminance / blackLuminance
    // targets = [whiteContrast, blackContrast]
    // console.log(targets)
    // return whiteContrast > blackContrast ?  [1, 0] : [0, 1]
    total = rgb[0] + rgb[1] + rgb[2]
    if (total >= 1.35) {
        return [0, 1]
    } else {
        return [1, 0]
    }
}

function luminance(rgb) {
    for (let i = 0; i < rgb.length; i++) {
        if (rgb[i] <= 0.03928) {
            rgb[i] /= 12.92
        } else {
            rgb[i] = pow((rgb[i]+0.055)/1.055, 2.4)
        }
    } 
    let lum = (0.2126 * rgb[0]) + (0.7152 * rgb[1]) + (0.0722 * rgb[2]) + 0.05
    return lum
}
