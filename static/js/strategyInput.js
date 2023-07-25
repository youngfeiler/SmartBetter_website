$(document).ready(function() {
  // Submit form using AJAX
  $("#input-form").submit(function(event) {
    event.preventDefault(); // Prevent form submission

    // Serialize form data
    var formData = $(this).serialize();

    // Send form data to Flask function
    $.ajax({
      url: "/make_strategy",
      type: "POST",
      data: formData,
      success: function(response) {
        // Handle response from Flask function
        console.log(response);
        if (response.status === 'error') {
          // Show the error message on the page
          alert(response.message); // You can use other methods to display the error message (e.g., showing it in a div)

        } else {
          // Redirect to the profile page after successful submission
          window.location.href = "/profile";
        }
      },
      error: function(xhr, status, error) {
        // Handle error, if any
        console.log("Error:", status, error);
        // Add your custom logic here
      }
    });
  });
});