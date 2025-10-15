document.addEventListener("DOMContentLoaded", function() {
    const tbody = document.querySelector("#sales-table tbody");
    const rows = Array.from(tbody.querySelectorAll("tr"));

    const rowsPerPage = 10; // показываем по 10 строк
    let currentPage = 1;
    const totalPages = Math.ceil(rows.length / rowsPerPage);

    const pageInfo = document.getElementById("page-info");
    const prevBtn = document.getElementById("prev-page");
    const nextBtn = document.getElementById("next-page");

    function renderPage(page) {
        rows.forEach(row => row.style.display = "none");

        const start = (page - 1) * rowsPerPage;
        const end = start + rowsPerPage;
        rows.slice(start, end).forEach(row => row.style.display = "");

        pageInfo.textContent = `Страница ${page} из ${totalPages}`;

        prevBtn.disabled = page === 1;
        nextBtn.disabled = page === totalPages;
    }

    prevBtn.addEventListener("click", () => {
        if (currentPage > 1) {
            currentPage--;
            renderPage(currentPage);
        }
    });

    nextBtn.addEventListener("click", () => {
        if (currentPage < totalPages) {
            currentPage++;
            renderPage(currentPage);
        }
    });

    renderPage(currentPage);
});
