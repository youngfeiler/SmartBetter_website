$(document).ready(function() {
    // When the element with ID "delete-form" is clicked
    $("#delete-form").on("click", function(event) {
        event.preventDefault(); // Prevent default behavior of the button

        // Show the user input popup
        showUserInputPopup();
    });

    // Function to show the user input popup
    function showUserInputPopup() {
        // Create a popup to ask for user input
        var userInput = prompt("Please enter your input:");

        // If the user clicks "Cancel" or enters an empty input, do nothing
        if (userInput === null || userInput.trim() === '') {
            return;
        }

        // Send the user input to the Flask function
        $.ajax({
            url: "/get_input",
            type: "POST",
            data: { user_input: userInput },
            success: function(response) {
                // Handle the success response
                console.log(response);

                // You can display a success message or perform other actions here
            },
            error: function(xhr, status, error) {
                // Handle error, if any
                console.log("Error:", status, error);
                // Add your custom error handling logic here
            }
        });
    }
});