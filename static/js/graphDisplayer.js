$(document).ready(function() {
  // Function to handle hover effect
  function handleTabHover(event) {
    $(this).toggleClass("hovered");
  }

  // Function to handle click event
  function handleTabClick(event) {
    // Remove active class from all tabs
    $(".tab").removeClass("active");
    // Set active class to the clicked tab
    $(this).addClass("active");
    const graphInfoEl = document.getElementById('graph-info-tab');
    graphInfoEl.click();
    const activeBetsEl = document.getElementById('active-bets-view');
    activeBetsEl.click();

    $('#day-info-tab').removeClass('active');
    // Call the Python function and pass the button text
    var strategy = $(this).text();
    $.ajax({
      url: "/get_graph_data",
      type: "POST",
      data: JSON.stringify({ strategy: strategy }),
      contentType: "application/json; charset=utf-8",
      dataType: "json",
      success: function(graphData) {
        if (graphData.status === 'error') {
          $('#graph').text(graphData.message).css('color', 'white');;
        } else {
        $('#graph').empty();
        var xData = [];
        var yData = [];
        var infoData = [];
        var dayResultInfo = [];
        var TotalPL;
        var totalPrecision;
        var bestDay;
        var worstDay;
        var totalBetsPlaced;
        var returnOnMoney;

        // Extract data for x, y, and info
        graphData.forEach(function(item) {
          xData.push(item.date);
          yData.push(item.result_sum);
          TotalPL = item.result_sum;
          totalPrecision = item.total_precision;
          bestDay = item.best_day;
          worstDay = item.worst_day;
          totalBetsPlaced = item.total_bets_placed;
          returnOnMoney = item.return_on_money;

          var teamsInfo = "";
          item.teams.forEach(function(team) {
            teamsInfo += team.team + ": " + team.result + "<br>";
          });

          var dayInfo = "";
          item.day_results.forEach(function(day) {
            dayInfo += "<b>Cumulative: $" + item.result_sum + "<br> <b>Day: $"+ day.daily_result;
          });
          infoData.push(teamsInfo);
          dayResultInfo.push(dayInfo);
        });

        var trace = {
          x: xData,
          y: yData,
          mode: 'lines+markers',
          hovertext: dayResultInfo,
          hoverinfo: 'text',
          marker: {
            size: 10,
            color : '#5ebde4',
          },
          hoverlabel: {
            bgcolor: '#124d70', // Customize background color of the hover template
            bordercolor: '#124d70', // Set the border color of the hover template
            font: {
              color: 'white', // Customize text color of the hover template
              size: 18, // Customize font size of the hover template
            },
          },
        };

        var layout = {
          title: 'Running P/L by Day',
          xaxis: {
            title: 'Date',
            linecolor: '#fefefe',
            tickfont: {
              color: '#fefefe',
            },
            automargin: true,
          },
          yaxis: {
            title: 'Running P/L'
          },
          font: {
            color: '#fefefe',
          },
          plot_bgcolor: '#16242f', // Change this to your desired background color
          // You can also set the color of the plot area's border using `paper_bgcolor`
          paper_bgcolor: '#16242f',
        };


        var graph = document.getElementById('graph');
        Plotly.newPlot(graph, [trace], layout);

        var stratNameEl = document.getElementById('strategy-name');
        stratNameEl.innerHTML = strategy;

        var overallStats = document.getElementById('overall-info-box');

        var worstDayEl = overallStats.querySelector('#worst-day');
        var bestDayEl = overallStats.querySelector('#best-day');
        var totalPlEl = overallStats.querySelector('#total-pl');
        var precEl = overallStats.querySelector('#precision');
        var totalBetsPlacedEl = overallStats.querySelector('#total-bets-placed');
        var returnOnMoneyEl = overallStats.querySelector('#return-on-money');

        worstDayEl.innerHTML = worstDay ;
        bestDayEl.innerHTML = "+"+bestDay;
        totalPlEl.innerHTML = "+"+TotalPL ;
        precEl.innerHTML = totalPrecision + "%";
        totalBetsPlacedEl.innerHTML =totalBetsPlaced;
        returnOnMoneyEl.innerHTML = returnOnMoney + "%";

        $('#graph-info-tab').addClass('active');

        // Rest of your success function code...

        // Event handler for clicking on a data point
        graph.on('plotly_click', function(data) {
          var pointIndex = data.points[0].pointIndex;
          var selectedInfo = infoData[pointIndex];
          var selectedDate = data.points[0].x; // Get the date associated with the clicked point


          // Show the day-info-list when a data point is clicked
          $('#info-list').hide();
          var infoBox = document.getElementById('day-info-list');
          infoBox.innerHTML = "<b>Date: " + selectedDate + "</b><br>" + selectedInfo;
          $('#day-info-list').show()
          $('#day-info-tab').addClass('active');
          $('#graph-info-tab').removeClass('active');
        });

      }}
    });
  }

  

  // // Event delegation for the hover effect
  $(".tabs-container").on("mouseenter", ".tab", handleTabHover).on("mouseleave", ".tab", handleTabHover);

  // Event delegation for the click effect
  $(".tabs-container").on("click", ".tab", handleTabClick);

  // // Show the default tab (Graph Information) on page load
  $('#graph-info-list').show();
  $('#day-info-list').hide();
  


  
  // Function to handle tab click event
  $('.tab.info').click(function() {

    $('.tab.info').removeClass('active');

    $(this).addClass('active');

    var tabId = $(this).attr('id');

    if (tabId === 'graph-info-tab') {
      $('#info-list').show();
      $('#day-info-list').hide();
    } else if (tabId === 'day-info-tab') {
      $('#info-list').hide();
      $('#day-info-list').show();
    }
  });

});
