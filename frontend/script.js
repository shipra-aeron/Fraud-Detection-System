document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('fileInput');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('http://localhost:5000/upload', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        document.getElementById('result').textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        console.error('There was a problem with the fetch operation:', error);
        document.getElementById('result').textContent = 'Error: ' + error.message;
    }
});
