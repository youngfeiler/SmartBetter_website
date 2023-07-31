$(document).ready(function() {
  $(".tabs-list").on("click", ".tab.strategy", function() {
    const strategy = $(this).text(); 
    $.ajax({
      type: "POST",
      url: "/check_if_text_allowed", 
      data: { strategy: strategy }, 
      success: function(response) {
        if (response.allowed) {
          $("#textMeCheckbox").prop("checked", true);
        } else {
          $("#textMeCheckbox").prop("checked", false);
        }
      },
      error: function(error) {
      }
    });
  });

  $("#textMeCheckbox").on("change", function() {

    const isChecked = $(this).prop("checked");

    const strategy = $('.tab.strategy.active').text();


    $.ajax({
      type: "POST",
      url: "/update_text_alert", 
      data: JSON.stringify({ isChecked: isChecked, strategy: strategy }),
      contentType: "application/json", 
      success: function(response) {
        console.log(''); 
      },
      error: function(error) {
        console.error("Error occurred:", error);
      }
    });
});


  const activeStrategy = $('.tab.strategy.active').text();
  $(".tab.strategy.active").trigger("click");
});
