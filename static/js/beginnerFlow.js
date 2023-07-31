const questions = document.querySelectorAll('.question');
let currentQuestionIndex = 0;
const selectedAnimals = new Set();

function showQuestion(index) {
    questions[currentQuestionIndex].classList.remove('active');
    questions[index].classList.add('active');
    currentQuestionIndex = index;
}

function selectOption(button) {
    const options = button.parentElement.querySelectorAll('button');
    options.forEach((option) => {
        option.classList.remove('selected');
    });
    button.classList.add('selected');
}

function nextQuestion() {
    if (currentQuestionIndex < questions.length - 1) {
        showQuestion(currentQuestionIndex + 1);
    }
}

function toggleAnimal(button, animal) {
    if (selectedAnimals.has(animal)) {
        selectedAnimals.delete(animal);
        button.classList.remove('selected');
    } else {
        selectedAnimals.add(animal);
        button.classList.add('selected');
    }
}

function submitForm() {
  const selectedOptions = document.querySelectorAll('.selected');
  const data = {};
  data['risk_tolerance'] = selectedOptions[0].innerText.trim();
  data['amount_of_bets'] = selectedOptions[1].innerText.trim();
  data['pre_game'] = selectedOptions[2].innerText.trim();
  data['text_allowed'] = selectedOptions[3].innerText.trim();
  data['strat_name'] = document.getElementById('nameInput').value;
  data['bettable_books'] = Array.from(selectedAnimals);

  fetch('/submit_beginner_strategy', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
  })
  .then((response) => {
    if (!response.ok) {
        return response.json().then((errorData) => {
            throw new Error(errorData.message);
        });
    }
    return response.json();
})
.then((data) => {
    console.log('Success');
})
.catch((error) => {
    alert(error.message);
});
}








