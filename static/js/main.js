// Main Client-side JS for EduPredict ML

document.addEventListener('DOMContentLoaded', () => {
    // Sync slider/number labels if present
    const attInput = document.getElementById('Attendance');
    const attVal = document.getElementById('att_val');
    if (attInput && attVal) {
        attInput.addEventListener('input', (e) => {
            attVal.textContent = e.target.value + '%';
        });
    }

    const hoursInput = document.getElementById('Hours_Studied');
    const hoursVal = document.getElementById('hours_val');
    if (hoursInput && hoursVal) {
        hoursInput.addEventListener('input', (e) => {
            hoursVal.textContent = e.target.value + ' hrs';
        });
    }
});
