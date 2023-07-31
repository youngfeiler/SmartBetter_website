$(document).ready(function() {
$.ajax({
  url: "/get_user_strategies",
  type: "GET",
  success: function(data) {
    // Process the received data
    data.forEach(function(strategy) {
      // Create buttons for each strategy
      var buttonId = strategy;
      var buttonLabel = strategy;

      var button = $('<li>').attr('id', buttonId).text(buttonLabel).addClass('tab strategy');
      button.appendTo('#tabs-container');
    });
    $("#tabs-container .tab:first-child").click();
  }
})});
