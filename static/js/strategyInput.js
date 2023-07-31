$(document).ready(function() {
  var selectedBooks = [];

  $(".bettable-books").on("click", function() {
    var bookName = $(this).val();
    $(this).toggleClass("selected");

    if ($(this).hasClass("selected")) {
      selectedBooks.push(bookName);
    } else {
      var index = selectedBooks.indexOf(bookName);
      if (index !== -1) {
        selectedBooks.splice(index, 1);
      }
    }

    // Update the value of the hidden input field with the selected books as a comma-separated string.
    $("#bettable_books").val(selectedBooks.join(','));
  });

  $("#input-form").submit(function(event) {
    event.preventDefault();
    var formData = new FormData(this);

    // Display the selectedBooks array in the console before making the AJAX request

    $.ajax({
      url: "/make_strategy",
      type: "POST",
      data: formData,
      processData: false,
      contentType: false,
      success: function(response) {
        if (response.status === "error") {
          $("#error-message").show();
        } else {
          $("#error-message").hide();
          window.location.href = "/profile";
        }
      },
      error: function(xhr, status, error) {
        console.log("Error:", status, error);
      }
    });
  });
});
