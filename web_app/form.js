function loader(){
    document.getElementById('check').disabled = true;
    var pResult = document.getElementById('text-result');
    const loaderSpan = document.createElement('span');
    loaderSpan.classList.add('loader');
    pResult.appendChild(loaderSpan);
}

function processMessage(message){
    document.querySelector('.loader').style.display = "none"
    document.getElementById('check').disabled = false;
    var pResult = document.getElementById('text-result');
    pResult.innerHTML = message;

}

function tooShortText(){
     document.querySelector('.text-too-short').style.display="block";
     document.getElementById("story").style.border="2px solid red";
}

function callModel(story){
    // Define the API URL
    const apiUrl = 'https://yaizaargudin.pythonanywhere.com//?story='+story;
 
    // Make a GET request
    fetch(apiUrl)
    .then(response => {
        if (!response.ok) {
        throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {

        processMessage(data.message);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
function getStory() {
    var story = document.getElementById("story").value;
    if(story.length > 10){
        loader();
        document.getElementById("story").style.border="0px";
        document.querySelector('.text-too-short').style.display="none";
        callModel(story)
    }else{
        tooShortText();
    }
  }

document.addEventListener('DOMContentLoaded', function(){
    document.getElementById("check").addEventListener("click", getStory);
    document.getElementById("story").addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
          event.preventDefault();
          getStory();
        }
      });

});
