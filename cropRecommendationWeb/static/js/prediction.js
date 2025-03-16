const cropItems = document.querySelectorAll('.crop-item');
    const hiddenInput = document.getElementById('selected_crop_input');
    const saveButton = document.getElementById('save_button');

    cropItems.forEach(item => {
        item.addEventListener('click', () => {
            // Highlight the selected crop
            cropItems.forEach(crop => crop.classList.remove('selected'));
            item.classList.add('selected');

            // Set the selected crop in the hidden input
            hiddenInput.value = item.getAttribute('data-crop');

            // Enable the save button
            saveButton.disabled = false;
        });
    });


    document.addEventListener('DOMContentLoaded', () => {
      const cropItems = document.querySelectorAll('.crop-item'); // All crop options
      const saveButton = document.getElementById('save_button'); // Save button
      const selectedCropInput = document.getElementById('selected_crop_input'); // Hidden input
  
      cropItems.forEach(item => {
        item.addEventListener('click', () => {
          // Remove 'selected' class from all crop items
          cropItems.forEach(crop => crop.classList.remove('selected'));
          
          // Add 'selected' class to the clicked item
          item.classList.add('selected');
  
          // Update the hidden input with the selected crop
          selectedCropInput.value = item.getAttribute('data-crop');
  
          // Enable and show the Save button
          saveButton.style.display = 'inline-block';
          saveButton.disabled = false;
        });
      });
    

  });

