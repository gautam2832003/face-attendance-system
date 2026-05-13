function showNotification(message, type = 'info') {
    const notification = document.getElementById('notification');
    const messageEl = document.getElementById('notification-message');
    const icon = notification.querySelector('.notification-icon i');

    notification.className = 'notification ' + type;
    messageEl.textContent = message;

    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        info: 'fa-info-circle'
    };
    icon.className = 'fas ' + (icons[type] || icons.info);

    notification.classList.add('show');

    clearTimeout(notification._timeout);
    notification._timeout = setTimeout(() => {
        notification.classList.remove('show');
    }, 4000);
}

function closeNotification() {
    const notification = document.getElementById('notification');
    notification.classList.remove('show');
}

function showLoading() {
    document.getElementById('loading-overlay').classList.add('active');
}

function hideLoading() {
    document.getElementById('loading-overlay').classList.remove('active');
}

document.addEventListener('DOMContentLoaded', () => {
    const inputs = document.querySelectorAll('.input-group input');
    inputs.forEach(input => {
        input.addEventListener('focus', function () {
            this.closest('.input-group').querySelector('label')?.style.setProperty('color', 'var(--accent)');
        });
        input.addEventListener('blur', function () {
            this.closest('.input-group').querySelector('label')?.style.setProperty('color', 'var(--text-secondary)');
        });
    });

    const cards = document.querySelectorAll('.role-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function () {
            this.style.transform = 'translateY(-8px)';
        });
        card.addEventListener('mouseleave', function () {
            this.style.transform = 'translateY(0)';
        });
    });
});

function confirmAction(message) {
    return confirm(message);
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard', 'success');
    }).catch(() => {
        showNotification('Failed to copy', 'error');
    });
}

function formatDate(dateStr) {
    if (!dateStr) return '';
    const d = new Date(dateStr);
    return d.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

function formatTime(timeStr) {
    if (!timeStr) return '';
    try {
        const [h, m] = timeStr.split(':');
        const hour = parseInt(h);
        const ampm = hour >= 12 ? 'PM' : 'AM';
        const h12 = hour % 12 || 12;
        return `${h12}:${m} ${ampm}`;
    } catch {
        return timeStr;
    }
}
