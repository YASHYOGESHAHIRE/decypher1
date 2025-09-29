// Shared Header JavaScript for Compliance Checker
(function() {
  'use strict';

  // Initialize header functionality when DOM is loaded
  document.addEventListener('DOMContentLoaded', function() {
    initializeHeader();
    initializeAuthModal();
    setActiveNavLink();
  });

  function initializeHeader() {
    // Header scroll behavior
    const header = document.getElementById('header');
    if (header) {
      const onScroll = () => {
        if (window.scrollY > 10) {
          header.classList.add('scrolled');
        } else {
          header.classList.remove('scrolled');
        }
      };
      document.addEventListener('scroll', onScroll);
      onScroll(); // Initial call
    }

    // Mobile menu toggle
    const menuToggle = document.getElementById('menuToggle');
    const mobileMenu = document.getElementById('mobileMenu');
    
    if (menuToggle && mobileMenu) {
      menuToggle.addEventListener('click', () => {
        const open = document.body.classList.toggle('mobile-open');
        menuToggle.setAttribute('aria-expanded', String(open));
        if (open) {
          mobileMenu.querySelector('a,button')?.focus();
        }
      });

      // Close mobile menu when clicking on links
      mobileMenu.querySelectorAll('a').forEach(a => {
        a.addEventListener('click', () => {
          document.body.classList.remove('mobile-open');
          menuToggle.setAttribute('aria-expanded', 'false');
        });
      });
    }
  }

  function initializeAuthModal() {
    const authModal = document.getElementById('auth-modal');
    if (!authModal) return;

    const openAuth = (mode = 'login') => {
      authModal.showModal();
      document.getElementById('login-form').hidden = (mode !== 'login');
      document.getElementById('signup-form').hidden = (mode !== 'signup');
      document.getElementById('auth-title').textContent = (mode === 'login' ? 'Welcome back' : 'Create your account');
      tabButtons.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.authTab === mode)));
    };

    // Open auth modal buttons
    document.querySelectorAll('[data-open-auth]').forEach(btn => {
      btn.addEventListener('click', () => openAuth(btn.dataset.openAuth));
    });

    // Auth tab switching
    const tabButtons = document.querySelectorAll('[data-auth-tab]');
    tabButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.authTab;
        document.getElementById('login-form').hidden = (mode !== 'login');
        document.getElementById('signup-form').hidden = (mode !== 'signup');
        tabButtons.forEach(b => b.setAttribute('aria-pressed', String(b === btn)));
        document.getElementById('auth-title').textContent = (mode === 'login' ? 'Welcome back' : 'Create your account');
      });
    });

    // Close modal when clicking outside
    authModal.addEventListener('click', (e) => {
      const r = authModal.getBoundingClientRect();
      if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) {
        authModal.close();
      }
    });
  }

  function setActiveNavLink() {
    // Set active navigation link based on current page
    const currentPage = window.location.pathname.split('/').pop() || 'landing.html';
    const navLinks = document.querySelectorAll('.nav-links a, .mobile-menu a');
    
    navLinks.forEach(link => {
      const href = link.getAttribute('href');
      if (href && (href === currentPage || href.includes(currentPage))) {
        link.setAttribute('aria-current', 'page');
        link.classList.add('active');
      } else {
        link.removeAttribute('aria-current');
        link.classList.remove('active');
      }
    });

    // Special case for landing page
    if (currentPage === 'landing.html' || currentPage === '') {
      const homeLinks = document.querySelectorAll('a[href*="landing.html"], a[href="#home"]');
      homeLinks.forEach(link => {
        link.setAttribute('aria-current', 'page');
        link.classList.add('active');
      });
    }
  }

  // Export functions for external use if needed
  window.ComplianceChecker = window.ComplianceChecker || {};
  window.ComplianceChecker.header = {
    setActiveNavLink: setActiveNavLink
  };
})();
