function activeBetClick() {
  const strategy = $('.tab.strategy.active').text();
  const tab = document.getElementById('active-bets-view');
  const tableContainer = document.getElementById('active-bets-table-container');
  const plotContainer = document.getElementById('plotContainer'); 

  plotContainer.style.display = 'none'; 
  tableContainer.style.display  = 'block';

  // Add or remove 'active' class from the tab to change its appearance
  // tab.classList.toggle('active', tableContainer.style.display === 'block');
  $(".tab.view").not(tab).removeClass("active");
  // $(this).addClass("active");

  // If the table container is displayed, fetch and populate the data
  if (tableContainer.style.display === 'block') {
    $.ajax({
      type: 'GET',
      url: '/active_bets',
      data: { strategy: strategy},
      success: function (response) {
        populateTable(response);
        console.log(response);
      },
      error: function (error) {
        console.error('Error fetching data:', error);
      }
    });
  }
}

function populateTable(data) {
  const tableBody = document.getElementById('active-bets-table-body');
  let tableHTML = '';

  if (data && Array.isArray(data)) {
    data.forEach(function (datapoint) {
      const sportsbook = datapoint.sportsbook[0].replace(/[\[\]']+/g, '');



      tableHTML += `
        <tr>
          <td>${datapoint.date}</td>
          <td>${datapoint.ev}%</td>
          <td><b>${datapoint.team} </b>vs ${datapoint.opponent}</td>
          <td>Moneyline</td>
          <td>${datapoint.highest_bettable_odds}</td>
          <td>${sportsbook}</td>
          
        </tr>
      `;
    });
  } else {
    console.error('Data is undefined or not an array.');
  }

  tableBody.innerHTML = tableHTML;
}

document.getElementById('active-bets-view').addEventListener('click', activeBetClick);