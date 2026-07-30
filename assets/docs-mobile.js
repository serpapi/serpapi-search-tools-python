(function () {
  "use strict";

  var mobile = window.matchMedia("(max-width: 991.98px)");

  function offset() {
    var meta = document.querySelector('meta[name="quarto:offset"]');
    return meta ? meta.content : "";
  }

  function isProjectHomepage() {
    var path = window.location.pathname;
    if (path.endsWith("/")) return true;
    if (!path.endsWith("/index.html")) return false;
    return !/(?:\/docs\/cookbook|\/user-guide|\/sdk-examples|\/reference)\/index\.html$/.test(
      path,
    );
  }

  function currentArea() {
    var path = window.location.pathname;
    if (path.indexOf("/docs/cookbook/") !== -1) return "cookbook";
    if (path.indexOf("/sdk-examples/") !== -1) return "examples";
    if (path.indexOf("/reference/") !== -1) return "reference";
    return "guide";
  }

  function addUserGuideNavbarLink() {
    var navbar = document.querySelector(
      "#navbarCollapse .navbar-nav.navbar-nav-scroll.me-auto",
    );
    if (!navbar) return;

    var links = navbar.querySelectorAll(".nav-link");
    for (var i = 0; i < links.length; i += 1) {
      if (links[i].textContent.trim() === "User Guide") return;
    }

    var item = document.createElement("li");
    item.className = "nav-item";
    var link = document.createElement("a");
    link.className = "nav-link";
    link.href = offset() + "index.html";
    link.setAttribute("data-gd-user-guide", "true");
    var label = document.createElement("span");
    label.className = "menu-text";
    label.textContent = "User Guide";
    link.appendChild(label);
    item.appendChild(link);

    if (currentArea() === "guide") {
      link.classList.add("active");
    }
    if (isProjectHomepage()) {
      link.setAttribute("aria-current", "page");
    }

    navbar.insertBefore(item, navbar.firstChild);
  }

  function addExploreLinks() {
    var overlay = document.querySelector(".gd-menu-overlay");
    if (!overlay) return;

    var list = overlay.querySelector(".gd-menu-list");
    if (!list || list.querySelector("[data-gd-explore]")) return;

    var destinations = [
      { area: "guide", label: "User Guide", path: "index.html" },
      {
        area: "cookbook",
        label: "Cookbook",
        path: "docs/cookbook/index.html",
      },
      {
        area: "examples",
        label: "SDK Examples",
        path: "sdk-examples/index.html",
      },
      {
        area: "reference",
        label: "API Reference",
        path: "reference/index.html",
      },
    ];
    var activeArea = currentArea();
    var section = document.createElement("li");
    section.className = "gd-menu-section";
    section.dataset.gdExplore = "true";
    section.setAttribute("aria-hidden", "true");
    section.textContent = "Explore";
    list.appendChild(section);

    destinations.forEach(function (destination) {
      if (destination.area === activeArea) return;

      var item = document.createElement("li");
      item.dataset.gdExplore = "true";
      var link = document.createElement("a");
      link.className = "gd-menu-item";
      link.href = offset() + destination.path;
      var label = document.createElement("span");
      label.className = "gd-menu-item-label";
      label.textContent = destination.label;
      link.appendChild(label);
      item.appendChild(link);
      list.appendChild(item);
    });
  }

  function restoreHeaderAfterClose() {
    var open = document.body.classList.contains("gd-menu-open");
    var toggle = document.querySelector("body.gd-project-home .navbar-toggler");
    if (toggle) toggle.setAttribute("aria-expanded", open ? "true" : "false");
    if (open || !mobile.matches) return;

    var header = document.getElementById("quarto-header");
    if (!header) return;
    header.classList.remove("headroom--unpinned");
    header.classList.add("headroom--pinned");
  }

  function interceptHomepageMenu(event) {
    if (!mobile.matches || !window.__gdMenu) return;

    event.preventDefault();
    event.stopImmediatePropagation();
    if (window.__gdMenu.isOpen()) {
      window.__gdMenu.hide();
    } else {
      window.__gdMenu.show();
    }
  }

  function init() {
    addUserGuideNavbarLink();

    if (isProjectHomepage()) {
      document.body.classList.add("gd-project-home");
      var toggle = document.querySelector(".navbar-toggler");
      if (toggle) {
        toggle.removeAttribute("data-bs-toggle");
        toggle.removeAttribute("data-bs-target");
        toggle.removeAttribute("onclick");
        toggle.addEventListener("click", interceptHomepageMenu, true);
      }
    }

    var bodyClassObserver = new MutationObserver(function () {
      restoreHeaderAfterClose();
    });
    bodyClassObserver.observe(document.body, {
      attributes: true,
      attributeFilter: ["class"],
    });

    var overlayObserver = new MutationObserver(function () {
      addExploreLinks();
    });
    overlayObserver.observe(document.body, {
      childList: true,
      subtree: true,
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
