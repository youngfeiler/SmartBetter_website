const tabs = document.querySelectorAll(".tab"); 

// Add event listeners 
tabs.forEach(tab => {

  tab.addEventListener("mouseover", () => {
    tab.classList.add("hovered");
  });

  tab.addEventListener("mouseout", () => {  
    tab.classList.remove("hovered");
  });

  tab.addEventListener("click", () => {
    tabs.forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
  });

});