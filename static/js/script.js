const inputSteps = document.querySelectorAll('.input-step');
const nextButtons = document.querySelectorAll('.next-button');

let currentStep = 0;

function showStep(step) {
  inputSteps.forEach((stepElement, index) => {
    if (index === step) {
      stepElement.classList.add('visible');
    } else {
      stepElement.classList.remove('visible');
    }
  });
}

nextButtons.forEach((button) => {
  button.addEventListener('click', (event) => {
    event.preventDefault();
    currentStep++;
    if (currentStep >= inputSteps.length) {
      currentStep = inputSteps.length - 1;
    }
    showStep(currentStep);
  });
});

showStep(currentStep);
