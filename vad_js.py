

js = """
async function main() {
  const script1 = document.createElement("script");
  script1.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.js";
  document.head.appendChild(script1)
  const script2 = document.createElement("script");
  script2.onload = async () =>  {
    console.log("vad loaded") ;
    var record = document.querySelector('.record-button');
    record.textContent = "Just Start Talking!"
    record.style = "width: fit-content; padding-right: 0.5vw;"
    const myvad = await vad.MicVAD.new({
      model: "v5",
      positiveSpeechThreshold: 0.3,
      negativeSpeechThreshold: 0.3,
      minSpeechFrames: 10,
      preSpeechPadFrames: 150,
      onSpeechStart: () => {
        console.log("Speech start detected")
        var record = document.querySelector('.record-button');
        var play_button = document.getElementById("streaming_out").querySelector(".play-pause-button")
        var playing = play_button && (play_button.ariaLabel === "Pause");
        if (record != null && !playing) {
          console.log(record);
          record.click();
        }
      },
      onSpeechEnd: (audio) => {
        console.log("Speech end detected")
        var stop = document.querySelector('.stop-button');
        if (stop != null) {
          console.log(stop);
          stop.click();
        }
      }
    })
    myvad.start()
  }
  script2.src = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.22/dist/bundle.min.js";
  script1.onload = () =>  {
    console.log("onnx loaded") 
    document.head.appendChild(script2)
  };
}
"""

js_reset = """
() => {
  var record = document.querySelector('.record-button');
  record.textContent = "Just Start Talking!"
  record.style = "width: fit-content; padding-right: 0.5vw;"
}
"""