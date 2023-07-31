function generateGraph() {
  const strategy = $('.tab.strategy.active').text();
  const tab = document.getElementById('teams-view');
  const tableContainer = document.getElementById('active-bets-table-container');
  const plotContainer = document.getElementById('plotContainer');
  tableContainer.style.display = 'none';
  plotContainer.style.display = 'block';
  tab.classList.add('active');
  $(".tab.view").not(tab).removeClass("active");



  $.ajax({
      url: '/team_dist_data',
      type: 'GET',
      data: { strategy: strategy},
      dataType: 'json',
      success: function (data) {
          console.log(data)
          const { teams, above_zero_counts, below_zero_counts } = data;

          // // Create the Plotly bar graph data
          // Create the Plotly stacked bar graph data
      const graphData = [{
        x: teams,
        y: above_zero_counts,
        type: 'bar',

        marker: {
          color: '#5ebde4' // Color for bars above zero
        },
        name: 'Wins'
      },
      {
        x: teams,
        y: below_zero_counts,
        type: 'bar',
        marker: {
          color: '#253c4d' // Color for bars below zero
        },
        name: 'Losses'
      }];

      // Set the layout for the graph
      const layout = {
        title: 'Historical win rates broken down by team',
        xaxis: {
          title: 'Team',
          tickfont: {
            color: '#fefefe',
          },
          automargin: true,
        },
        yaxis: {
          title: 'Count',  
        },
        font: {
          color: '#fefefe',
        },
        barmode: 'relative',
        plot_bgcolor: '#16242f',
        paper_bgcolor: '#16242f',
      };
          Plotly.newPlot('plotContainer', graphData, layout);
      },
      error: function (error) {
          console.error('Error fetching data:', error);
      }
  });
}

document.getElementById('teams-view').addEventListener('click', generateGraph);
