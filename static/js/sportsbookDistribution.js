function generateGraph() {
  const strategy = $('.tab.strategy.active').text();
  const tab = document.getElementById('book-view');
  const tableContainer = document.getElementById('active-bets-table-container');
  const plotContainer = document.getElementById('plotContainer');
  tableContainer.style.display = 'none';
  plotContainer.style.display = 'block';
  tab.classList.add('active');
  
  $(".tab.view").not(tab).removeClass("active");

  $.ajax({
    url: '/book_dist_data',
    type: 'GET',
    data: { strategy: strategy },
    dataType: 'json',
    success: function (data) {
      console.log(data);
      const { book, above_zero_counts, below_zero_counts } = data;

      // Flatten the lists of 'sportsbook(s)_used' to show each individual item as x value
      const flattenedBook = book.flatMap((item) => item);

      // Flatten the corresponding counts accordingly
      const flattenedAboveZeroCounts = above_zero_counts.flatMap((item) => item);
      const flattenedBelowZeroCounts = below_zero_counts.flatMap((item) => item);

      // Create the Plotly stacked bar graph data
      const graphData = [
        {
          x: flattenedBook,
          y: flattenedAboveZeroCounts,
          type: 'bar',
          marker: {
            color: '#5ebde4' // Color for bars above zero
          },
          name: 'Wins'
        },
        {
          x: flattenedBook,
          y: flattenedBelowZeroCounts,
          type: 'bar',
          marker: {
            color: '#253c4d' // Color for bars below zero
          },
          name: 'Losses'
        }
      ];

      // Set the layout for the graph
      const layout = {
        title: 'Sportsbooks Used',
        xaxis: {
          title: 'Win rate history across different sportsbooks',
          tickfont: {
            color: '#fefefe',
          },
          automargin: true,
        },
        yaxis: {
          title: 'Count'
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

document.getElementById('book-view').addEventListener('click', generateGraph);
