/**
 * OccasionAI — Frontend Script
 * Multi-View Single-Page Application (SPA) supporting:
 *  - Discover Gifts (AI Neural Hybrid Vector Recommendation Search)
 *  - Admin Catalog Management (Product list table)
 *  - Add Product (Instant vector embedding generation & Qdrant/FAISS store)
 *  - 404 / API Disconnection Error Handling
 */

const API_BASE = "http://localhost:8000";
const DEFAULT_LIMIT = 12;

// ── DOM References ────────────────────────────────────────────────────────────
const adminToggleBtn = document.getElementById("admin-toggle-btn");
const brandHomeBtn   = document.getElementById("brand-home-btn");
const viewSections   = document.querySelectorAll(".view-section");

let currentView = "discover";

// View 1: Discover
const input         = document.getElementById("occasion-input");
const searchBtn     = document.getElementById("search-btn");
const productGrid   = document.getElementById("product-grid");
const loadingGrid   = document.getElementById("loading-grid");
const emptyState    = document.getElementById("empty-state");
const resultsHdr    = document.getElementById("results-header");
const resultsCount  = document.getElementById("results-count");
const resultsOcc    = document.getElementById("results-occasion-label");
const chips         = document.querySelectorAll(".chip");

// View 2: Admin Catalog
const adminCount    = document.getElementById("admin-product-count");
const adminTable    = document.getElementById("admin-table-body");
const goAddBtn      = document.getElementById("go-add-product-btn");
const adminSearch   = document.getElementById("admin-search-input");
const adminPrevBtn  = document.getElementById("admin-prev-page");
const adminNextBtn  = document.getElementById("admin-next-page");
const adminPageInfo = document.getElementById("admin-page-info");

let adminProducts = [];
let adminCurrentPage = 1;
const ADMIN_PAGE_SIZE = 10;

// View 3: Add Product Form
const addForm       = document.getElementById("add-product-form");
const submitBtn     = document.getElementById("submit-product-btn");
const formAlert     = document.getElementById("form-alert");

// View 4: Error Page
const retryBtn      = document.getElementById("retry-conn-btn");

// Footer year
document.getElementById("year").textContent = new Date().getFullYear();

// ── SPA View Switcher ─────────────────────────────────────────────────────────
function switchView(targetViewId) {
  currentView = targetViewId;

  viewSections.forEach(sec => {
    if (sec.id === `view-${targetViewId}`) {
      sec.hidden = false;
      sec.classList.add("active");
    } else {
      sec.hidden = true;
      sec.classList.remove("active");
    }
  });

  if (adminToggleBtn) {
    if (targetViewId === "discover") {
      adminToggleBtn.classList.remove("admin-mode");
      adminToggleBtn.innerHTML = `<span class="toggle-icon">🛡️</span><span class="toggle-text">Switch to Admin Portal</span>`;
    } else {
      adminToggleBtn.classList.add("admin-mode");
      adminToggleBtn.innerHTML = `<span class="toggle-icon">✦</span><span class="toggle-text">Back to Gift Finder</span>`;
    }
  }

  if (targetViewId === "admin") {
    loadAdminCatalog();
  }
}

if (adminToggleBtn) {
  adminToggleBtn.addEventListener("click", () => {
    if (currentView === "discover") {
      switchView("admin");
    } else {
      switchView("discover");
    }
  });
}

if (brandHomeBtn) {
  brandHomeBtn.addEventListener("click", () => switchView("discover"));
}

if (goAddBtn) {
  goAddBtn.addEventListener("click", () => switchView("add-product"));
}

if (retryBtn) {
  retryBtn.addEventListener("click", () => switchView("discover"));
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function inferCategory(product) {
  const text = `${product.name} ${product.brand || ""}`.toLowerCase();
  if (/flower|rose|lily|marigold|tulip|bouquet|floral/i.test(text))   return "🌸 Flowers";
  if (/chocolate|choco|cocoa|candy/i.test(text))                       return "🍫 Chocolates";
  if (/watch|timepiece/i.test(text))                                    return "⌚ Accessories";
  if (/perfume|fragrance|eau de|scent|aroma/i.test(text))              return "🌿 Fragrance";
  if (/dry fruit|almond|cashew|pistachio|hamper/i.test(text))          return "🥜 Hampers";
  if (/jewel|ring|necklace|bracelet|earring/i.test(text))              return "💎 Jewellery";
  if (/t.shirt|shirt|cloth|apparel|print|fabric/i.test(text))          return "👕 Apparel";
  if (/bamboo|eco|organic|sustainable/i.test(text))                    return "🌿 Eco Gifts";
  if (/self.care|wellness|spa|kit/i.test(text))                         return "✨ Self-Care";
  if (/cake|sweet|mithai|ladoo/i.test(text))                            return "🎂 Sweets";
  return "🛍️ Gift";
}

function formatPrice(price) {
  if (!price) return "₹999";
  if (String(price).startsWith("$") || String(price).startsWith("₹")) return price;
  const n = parseFloat(price);
  return isNaN(n) ? price : `₹${n.toLocaleString("en-IN")}`;
}

// ── View 1: Discover Gift Recommendation Cards ────────────────────────────────
function buildCard(product, index) {
  const card = document.createElement("article");
  card.className = "product-card";
  card.style.animationDelay = `${index * 0.04}s`;

  const imageWrap = document.createElement("div");
  imageWrap.className = "card-image-wrap";

  if (product.mainImage && product.mainImage !== "") {
    const img = document.createElement("img");
    img.className = "card-image";
    img.src = product.mainImage;
    img.alt = product.name;
    img.onerror = () => {
      // Revert back to the graceful DOM text placeholder
      imageWrap.innerHTML = `<div class="card-image-placeholder">${inferCategory(product).split(" ")[0]}</div>`;
    };
    imageWrap.appendChild(img);
  } else {
    // If no image provided at all, use default
    imageWrap.innerHTML = `<img src="https://images.unsplash.com/photo-1549465220-1a8b9238cd48?w=500&auto=format&fit=crop" class="card-image" alt="${product.name}" />`;
  }

  if (product.relevance_score != null) {
    const scoreBadge = document.createElement("span");
    scoreBadge.className = "score-badge";
    scoreBadge.textContent = `★ ${(product.relevance_score * 100).toFixed(0)}% Match`;
    imageWrap.appendChild(scoreBadge);
  }

  const body = document.createElement("div");
  body.className = "card-body";

  const category = document.createElement("div");
  category.className = "card-category";
  category.textContent = inferCategory(product);

  const name = document.createElement("h3");
  name.className = "card-name";
  name.textContent = product.name;

  const brand = document.createElement("div");
  brand.className = "card-brand";
  brand.textContent = product.brand ? `by ${product.brand}` : "";

  const desc = document.createElement("p");
  desc.className = "card-desc";
  desc.textContent = product.shortDescription || product.description || "High quality occasion recommendation.";

  const footer = document.createElement("div");
  footer.className = "card-footer";
  footer.innerHTML = `<span class="card-price">${formatPrice(product.price)}</span><span class="card-status status-active">Available</span>`;

  body.append(category, name, brand, desc, footer);
  card.append(imageWrap, body);
  return card;
}

function showEmpty(occasion) {
  loadingGrid.hidden = true;
  productGrid.hidden = true;
  emptyState.hidden  = false;
  resultsHdr.hidden  = true;

  const emptyTitle = document.getElementById("empty-title");
  const emptyDesc  = document.getElementById("empty-desc");

  if (occasion) {
    if (emptyTitle) emptyTitle.textContent = `No exact matches for "${occasion}"`;
    if (emptyDesc)  emptyDesc.innerHTML = `We couldn't find products matching <strong>"${occasion}"</strong>. Try picking a popular theme below:`;
  } else {
    if (emptyTitle) emptyTitle.textContent = "Ready to Find the Best Gift?";
    if (emptyDesc)  emptyDesc.innerHTML = `Enter an occasion above (like <strong>Birthday</strong>, <strong>Diwali</strong>, or <strong>Wedding</strong>) or pick a theme below.`;
  }
}

function showLoading() {
  if (!loadingGrid) return;
  loadingGrid.style.display = "grid";
  loadingGrid.hidden = false;
  loadingGrid.innerHTML = `
    <div class="skeleton-card"><div class="sk-img"></div><div class="sk-line l1"></div><div class="sk-line l2"></div></div>
    <div class="skeleton-card"><div class="sk-img"></div><div class="sk-line l1"></div><div class="sk-line l2"></div></div>
    <div class="skeleton-card"><div class="sk-img"></div><div class="sk-line l1"></div><div class="sk-line l2"></div></div>
  `;
  productGrid.hidden = true;
  emptyState.hidden  = true;
  resultsHdr.hidden  = true;
}

function hideLoading() {
  if (!loadingGrid) return;
  loadingGrid.style.display = "none";
  loadingGrid.hidden = true;
  loadingGrid.innerHTML = "";
}

async function fetchRecommendations(occasion) {
  if (!occasion.trim()) return;

  chips.forEach(c => c.classList.toggle("active", c.dataset.occasion === occasion));

  showLoading();

  try {
    const res = await fetch(`${API_BASE}/api/recommend?occasion=${encodeURIComponent(occasion.trim())}&limit=${DEFAULT_LIMIT}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();
    const products = data.recommendations || [];

    hideLoading();

    if (!products.length) {
      emptyState.hidden = false;
    } else {
      productGrid.hidden = false;
      resultsHdr.hidden  = false;
      resultsCount.textContent = `${products.length} Products Found`;
      resultsOcc.textContent   = `Occasion Query: "${occasion}"`;

      productGrid.innerHTML = "";
      products.forEach((p, i) => productGrid.appendChild(buildCard(p, i)));
    }
  } catch (err) {
    console.error("API error:", err);
    hideLoading();
    switchView("error");
  }
}

// Search Listeners
searchBtn.addEventListener("click", () => fetchRecommendations(input.value));
input.addEventListener("keydown", (e) => { if (e.key === "Enter") fetchRecommendations(input.value); });
chips.forEach(c => c.addEventListener("click", () => {
  input.value = c.dataset.occasion;
  fetchRecommendations(c.dataset.occasion);
}));

// ── View 2: Load Admin Catalog Products Table ─────────────────────────────────
async function loadAdminCatalog() {
  adminTable.innerHTML = `<tr><td colspan="6" style="text-align:center; padding: 2rem;">Loading products from vector database...</td></tr>`;
  try {
    const res = await fetch(`${API_BASE}/api/products?limit=500`);
    if (!res.ok) throw new Error("Failed to load products");
    adminProducts = await res.json();
    adminCurrentPage = 1;
    if(adminSearch) adminSearch.value = "";
    
    renderAdminTable();
  } catch (err) {
    console.error("Admin catalog load error:", err);
    adminTable.innerHTML = `<tr><td colspan="6" style="text-align:center; color:#f43f5e; padding: 2rem;">Failed to load catalog. Backend error.</td></tr>`;
  }
}

function renderAdminTable() {
  const searchTerm = adminSearch ? adminSearch.value.trim().toLowerCase() : "";
  const filtered = adminProducts.filter(p => p.name.toLowerCase().includes(searchTerm) || (p.brand && p.brand.toLowerCase().includes(searchTerm)));
  
  adminCount.textContent = filtered.length;
  
  const totalPages = Math.ceil(filtered.length / ADMIN_PAGE_SIZE) || 1;
  if(adminCurrentPage > totalPages) adminCurrentPage = totalPages;
  if(adminCurrentPage < 1) adminCurrentPage = 1;

  const startIdx = (adminCurrentPage - 1) * ADMIN_PAGE_SIZE;
  const paginated = filtered.slice(startIdx, startIdx + ADMIN_PAGE_SIZE);

  adminTable.innerHTML = "";
  
  if(paginated.length === 0) {
    adminTable.innerHTML = `<tr><td colspan="6" style="text-align:center; padding: 2rem;">No products found.</td></tr>`;
  } else {
    paginated.forEach(p => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><img src="${p.mainImage || 'https://images.unsplash.com/photo-1549465220-1a8b9238cd48?w=100'}" onerror="this.onerror=null; this.src='https://via.placeholder.com/100x100?text=Error';" class="table-thumb" alt="${p.name}" /></td>
        <td><strong>${p.name}</strong></td>
        <td>${p.brand || 'Generic'}</td>
        <td>${formatPrice(p.price)}</td>
        <td>${p.stock ?? 50}</td>
        <td><span class="card-status status-active">ACTIVE</span></td>
      `;
      adminTable.appendChild(tr);
    });
  }

  // Update pagination controls
  if(adminPageInfo) adminPageInfo.textContent = `Page ${adminCurrentPage} of ${totalPages}`;
  if(adminPrevBtn) adminPrevBtn.disabled = adminCurrentPage === 1;
  if(adminNextBtn) adminNextBtn.disabled = adminCurrentPage === totalPages;
}

if(adminSearch) {
  adminSearch.addEventListener("input", () => {
    adminCurrentPage = 1;
    renderAdminTable();
  });
}
if(adminPrevBtn) {
  adminPrevBtn.addEventListener("click", () => {
    if(adminCurrentPage > 1) { adminCurrentPage--; renderAdminTable(); }
  });
}
if(adminNextBtn) {
  adminNextBtn.addEventListener("click", () => {
    adminCurrentPage++; renderAdminTable();
  });
}

// ── View 3: Add Product Form Submission ───────────────────────────────────────
addForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  
  const submitBtnSpan = submitBtn.querySelector("span");
  submitBtn.disabled = true;
  submitBtnSpan.textContent = "⏳ Encoding Vector Embedding...";

  formAlert.hidden = true;
  formAlert.className = "alert-box";

  const newProduct = {
    name: document.getElementById("p-name").value.trim(),
    brand: document.getElementById("p-brand").value.trim() || "Generic",
    price: document.getElementById("p-price").value.trim() || "$29.99",
    mainImage: document.getElementById("p-image").value.trim() || "https://images.unsplash.com/photo-1549465220-1a8b9238cd48?w=500&auto=format&fit=crop",
    description: document.getElementById("p-desc").value.trim() || document.getElementById("p-name").value.trim(),
    shortDescription: document.getElementById("p-desc").value.trim(),
    stock: 50,
  };

  try {
    const res = await fetch(`${API_BASE}/api/products`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(newProduct),
    });

    if (!res.ok) throw new Error("Failed to add product");

    const created = await res.json();

    formAlert.hidden = false;
    formAlert.classList.add("success");
    formAlert.innerHTML = `✅ <strong>Product Added & Indexed!</strong> Vector embedding generated for <em>"${created.name}"</em>.`;

    addForm.reset();
    setTimeout(() => {
      switchView("admin");
    }, 1800);

  } catch (err) {
    console.error("Add product error:", err);
    formAlert.hidden = false;
    formAlert.classList.add("error");
    formAlert.textContent = `⚠️ Failed to save product. Make sure backend is running on ${API_BASE}.`;
  } finally {
    submitBtn.disabled = false;
    submitBtnSpan.textContent = "✨ Save & Generate Vector Embedding";
  }
});

// ── Initial Load ──────────────────────────────────────────────────────────────
window.addEventListener("load", () => {
  input.value = "Birthday";
  fetchRecommendations("Birthday");
});
