/**
 * Photobomb Studio — Premium Interactive Frontend Engine
 * Architected by Senior Staff UI/UX & Frontend Engineer
 */

let currentDetectionData = null;
let currentViewMode = "annotated"; // 'annotated' | 'original' | 'cleaned'
let hoveredPersonId = null;
let sampleImagesList = [];
let currentCategoryFilter = "all";

// Multi-Photo Batch Session State (Up to 15+ concurrent photos)
let batchPhotos = [];
let activePhotoId = null;

function getActivePhoto() {
  return batchPhotos.find((p) => p.id === activePhotoId) || null;
}

const DEFAULT_WEIGHTS = {
  centrality: 25,
  size: 25,
  focus: 20,
  depth: 15,
  pose: 15,
};

// DOM Cache
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const workspaceSection = document.getElementById("workspace-section");
const imageDisplay = document.getElementById("image-display");
const interactiveCanvas = document.getElementById("interactive-canvas");
const canvasContainer = document.getElementById("canvas-container");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingText = document.getElementById("loading-text");
const jsonPreBlock = document.getElementById("json-pre-block");
const personsList = document.getElementById("persons-list");
const sampleCardsGrid = document.getElementById("sample-cards-grid");

// App Bootstrap
document.addEventListener("DOMContentLoaded", () => {
  setupUploadAndPasteListeners();
  setupCanvasListeners();
  fetchSampleImages();
  fetchEvaluationBenchmarkOnLoad();
});

// Setup File Drag, Drop & Clipboard Paste (Cmd+V) with Multi-File Batch Support
function setupUploadAndPasteListeners() {
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("border-indigo-500", "bg-indigo-950/20", "scale-[1.01]");
  });

  dropZone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dropZone.classList.remove("border-indigo-500", "bg-indigo-950/20", "scale-[1.01]");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("border-indigo-500", "bg-indigo-950/20", "scale-[1.01]");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleIncomingFiles(e.dataTransfer.files);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleIncomingFiles(e.target.files);
      fileInput.value = "";
    }
  });

  const batchAddInput = document.getElementById("batch-add-input");
  if (batchAddInput) {
    batchAddInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleIncomingFiles(e.target.files);
        batchAddInput.value = "";
      }
    });
  }

  // Global Clipboard Paste Support (Cmd+V / Ctrl+V)
  window.addEventListener("paste", (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    const pasted = [];
    for (const item of items) {
      if (item.type.indexOf("image") !== -1) {
        const blob = item.getAsFile();
        if (blob) pasted.push(blob);
      }
    }
    if (pasted.length > 0) {
      showToast(`Pasted ${pasted.length} image(s) from clipboard 📋`);
      handleIncomingFiles(pasted);
    }
  });
}

// Fetch Sample Gallery from Backend
async function fetchSampleImages() {
  try {
    const res = await fetch("/api/samples");
    const data = await res.json();
    sampleImagesList = data.samples || [];

    renderSampleCards();

    // Auto-load a prominent photobomb sample on initial launch
    loadSampleImageById("pb_02");
  } catch (err) {
    console.error("Failed to fetch sample gallery:", err);
  }
}

function filterSampleCategory(cat) {
  currentCategoryFilter = cat;
  ["all", "photobomber", "ambiguous", "clean"].forEach((c) => {
    const btn = document.getElementById(`filter-${c}`);
    if (c === cat) {
      btn.className = "px-2.5 py-1 rounded-lg font-semibold bg-indigo-600 text-white shadow-sm transition-all";
    } else {
      btn.className = "px-2.5 py-1 rounded-lg font-semibold text-zinc-400 hover:text-white transition-all";
    }
  });
  renderSampleCards();
}

function renderSampleCards() {
  if (!sampleCardsGrid) return;

  const filtered = sampleImagesList.filter((s) => {
    if (currentCategoryFilter === "all") return true;
    return s.category === currentCategoryFilter;
  });

  const categoryBadges = {
    photobomber: { label: "Photobomb", bg: "bg-rose-500/10 text-rose-400 border-rose-500/20" },
    ambiguous: { label: "Crowded", bg: "bg-amber-500/10 text-amber-400 border-amber-500/20" },
    clean: { label: "Clean", bg: "bg-sky-500/10 text-sky-400 border-sky-500/20" },
  };

  sampleCardsGrid.innerHTML = filtered
    .slice(0, 10)
    .map((s) => {
      const b = categoryBadges[s.category] || { label: "Sample", bg: "bg-zinc-800 text-zinc-300 border-white/5" };
      return `
      <div onclick="loadSampleImageById('${s.id}')" 
           class="group relative overflow-hidden rounded-xl bg-zinc-900/60 border border-white/[0.08] hover:border-indigo-500/50 p-2 cursor-pointer transition-all duration-200 hover:-translate-y-1 hover:shadow-xl">
        <div class="aspect-[4/3] rounded-lg overflow-hidden bg-black/40 mb-2 relative">
          <img src="${s.url}" alt="${s.filename}" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300">
          <div class="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent"></div>
          <span class="absolute bottom-1.5 left-1.5 text-[9px] font-bold px-1.5 py-0.5 rounded border uppercase tracking-wider backdrop-blur-sm ${b.bg}">
            ${b.label}
          </span>
        </div>
        <div class="flex items-center justify-between text-xs">
          <span class="font-bold text-white truncate text-[11px]">${s.filename}</span>
          <span class="text-indigo-400 opacity-0 group-hover:opacity-100 transition-opacity text-xs">&rarr;</span>
        </div>
      </div>
    `;
    })
    .join("");
}

// Load a specific sample image into batch session
async function loadSampleImageById(sampleId) {
  const sample = sampleImagesList.find((s) => s.id === sampleId);
  if (!sample) return;

  showLoading(`Loading ${sample.filename}...`);
  try {
    const imgRes = await fetch(sample.url);
    const blob = await imgRes.blob();
    const file = new File([blob], sample.filename, { type: "image/jpeg" });
    await handleIncomingFiles([file]);
    showToast(`Loaded ${sample.filename}`);
  } catch (err) {
    hideLoading();
    showToast("Error loading sample: " + err.message, "error");
  }
}

// Ingest multiple incoming files (supporting 15+ concurrent uploads)
async function handleIncomingFiles(fileList) {
  if (!fileList || fileList.length === 0) return;

  const validFiles = Array.from(fileList).filter((f) => f.type && f.type.startsWith("image/"));
  if (validFiles.length === 0) {
    showToast("Please provide valid image files (JPEG, PNG, WEBP)", "error");
    return;
  }

  const wasEmpty = batchPhotos.length === 0;

  const newItems = validFiles.map((file, idx) => ({
    id: "photo_" + Date.now() + "_" + idx + "_" + Math.random().toString(36).substr(2, 6),
    file: file,
    filename: file.name,
    previewUrl: URL.createObjectURL(file),
    status: "pending", // 'pending' | 'analyzing' | 'ready' | 'cleaned' | 'error'
    detectionData: null,
    cleanedImage: null,
    viewMode: "annotated",
    threshold: parseFloat(document.getElementById("threshold-slider") ? document.getElementById("threshold-slider").value : "0.62"),
    error: null,
  }));

  batchPhotos.push(...newItems);
  renderBatchScrollBar();

  showToast(`Added ${newItems.length} photo(s) to session 📸`);

  // Activate the first newly added photo if nothing active or session was empty
  if (wasEmpty || !activePhotoId) {
    await selectBatchPhoto(newItems[0].id);
  }
}

// Render the interactive horizontal scroll bar of batch photos
function renderBatchScrollBar() {
  const container = document.getElementById("batch-gallery-container");
  const scrollBar = document.getElementById("batch-scroll-bar");
  const countBadge = document.getElementById("batch-count-badge");
  if (!container || !scrollBar) return;

  if (batchPhotos.length === 0) {
    container.classList.add("hidden");
    return;
  }

  container.classList.remove("hidden");
  if (countBadge) {
    countBadge.textContent = `${batchPhotos.length} Photo${batchPhotos.length === 1 ? "" : "s"}`;
  }

  scrollBar.innerHTML = batchPhotos
    .map((photo, idx) => {
      const isActive = photo.id === activePhotoId;
      const isCleaned = !!photo.cleanedImage;
      const isAnalyzing = photo.status === "analyzing";
      const hasDetections = !!photo.detectionData;

      let badgeHtml = "";
      if (isCleaned) {
        badgeHtml = `<span class="absolute bottom-1 left-1 text-[9px] font-bold px-1.5 py-0.5 rounded bg-emerald-600/90 text-white shadow backdrop-blur-sm">✨ Cleaned</span>`;
      } else if (isAnalyzing) {
        badgeHtml = `<span class="absolute bottom-1 left-1 text-[9px] font-bold px-1.5 py-0.5 rounded bg-indigo-600/90 text-white animate-pulse shadow backdrop-blur-sm">⚡ Analyzing</span>`;
      } else if (hasDetections) {
        const count = photo.detectionData.num_persons_detected;
        badgeHtml = `<span class="absolute bottom-1 left-1 text-[9px] font-bold px-1.5 py-0.5 rounded bg-zinc-900/90 text-indigo-300 border border-indigo-500/30 shadow backdrop-blur-sm">${count} Pers</span>`;
      } else {
        badgeHtml = `<span class="absolute bottom-1 left-1 text-[9px] font-bold px-1.5 py-0.5 rounded bg-zinc-900/90 text-zinc-400 border border-white/10 shadow backdrop-blur-sm">Ready</span>`;
      }

      return `
      <div onclick="selectBatchPhoto('${photo.id}')"
           id="thumb-${photo.id}"
           class="batch-thumb-card flex-shrink-0 w-36 sm:w-40 rounded-xl p-2 cursor-pointer relative group ${isActive ? "batch-thumb-active" : ""}">
        <!-- Delete Button (X) -->
        <button onclick="deleteBatchPhoto('${photo.id}', event)"
                class="batch-delete-btn absolute top-1.5 right-1.5 w-5 h-5 rounded-full bg-zinc-900/90 hover:bg-rose-600 text-zinc-300 hover:text-white flex items-center justify-center text-xs shadow-md border border-white/15 z-20 cursor-pointer transition-transform"
                title="Delete this photo from session">
          &times;
        </button>

        <!-- Thumbnail Image -->
        <div class="w-full h-20 rounded-lg overflow-hidden bg-black/50 relative mb-1.5 flex items-center justify-center">
          <img src="${photo.previewUrl}" alt="${photo.filename}" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200">
          <div class="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent pointer-events-none"></div>
          ${badgeHtml}
        </div>

        <!-- Meta -->
        <div class="px-0.5">
          <span class="text-[11px] font-bold text-white truncate block ${isActive ? "text-indigo-300" : ""}" title="${photo.filename}">
            ${idx + 1}. ${photo.filename}
          </span>
          <div class="flex items-center justify-between text-[10px] text-zinc-400 mt-0.5">
            <span>${isCleaned ? "Erased" : (hasDetections ? "Analyzed" : "Click to edit")}</span>
            ${isActive ? '<span class="text-indigo-400 font-bold text-[9px] uppercase tracking-wider">Active</span>' : ''}
          </div>
        </div>
      </div>
      `;
    })
    .join("");

  // Smoothly scroll active card into view if needed
  if (activePhotoId) {
    const activeEl = document.getElementById(`thumb-${activePhotoId}`);
    if (activeEl) {
      activeEl.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
    }
  }
}

// Select active photo from the scroll bar
async function selectBatchPhoto(photoId) {
  const photo = batchPhotos.find((p) => p.id === photoId);
  if (!photo) return;

  if (activePhotoId === photoId && currentDetectionData) {
    return; // Already loaded and active
  }

  activePhotoId = photoId;
  renderBatchScrollBar();

  workspaceSection.classList.remove("hidden");

  // If already analyzed, restore state instantaneously
  if (photo.detectionData) {
    currentDetectionData = photo.detectionData;
    currentViewMode = photo.viewMode || (photo.cleanedImage ? "cleaned" : "annotated");

    // Restore slider and threshold
    const slider = document.getElementById("threshold-slider");
    if (slider && photo.threshold !== undefined) {
      slider.value = photo.threshold;
      const valDisplay = document.getElementById("threshold-val");
      if (valDisplay) valDisplay.textContent = Number(photo.threshold).toFixed(2);
    }

    renderDetectionResults(photo.detectionData);

    // Update view mode buttons
    const btnCleaned = document.getElementById("btn-view-cleaned");
    const btnRevert = document.getElementById("btn-revert-original");
    if (photo.cleanedImage) {
      if (btnCleaned) btnCleaned.classList.remove("hidden");
      if (btnRevert) btnRevert.classList.remove("hidden");
    } else {
      if (btnCleaned) btnCleaned.classList.add("hidden");
      if (btnRevert) btnRevert.classList.add("hidden");
    }

    setViewMode(currentViewMode);
  } else {
    // Needs analysis
    await analyzeBatchPhoto(photo);
  }
}

// Run YOLOv8 detection & signal extraction for a specific batch photo
async function analyzeBatchPhoto(photo) {
  photo.status = "analyzing";
  renderBatchScrollBar();
  showLoading(`Analyzing ${photo.filename}...`);

  const formData = new FormData();
  formData.append("file", photo.file);
  const currentThreshold = photo.threshold !== undefined ? photo.threshold.toString() : (document.getElementById("threshold-slider") ? document.getElementById("threshold-slider").value : "0.62");
  formData.append("threshold", currentThreshold);
  formData.append("conf_threshold", "0.25");

  const weights = getNormalizedWeights();
  formData.append("weight_centrality", weights.centrality);
  formData.append("weight_size", weights.relative_size);
  formData.append("weight_focus", weights.focus);
  formData.append("weight_depth", weights.depth);
  formData.append("weight_pose", weights.pose);

  try {
    const res = await fetch("/api/detect", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Detection failed");
    }

    const data = await res.json();
    photo.detectionData = data;
    photo.status = "ready";
    photo.threshold = parseFloat(currentThreshold);
    photo.viewMode = "annotated";

    if (activePhotoId === photo.id) {
      currentDetectionData = data;
      renderDetectionResults(data);
      setViewMode("annotated");
      showToast(`Analyzed ${photo.filename} (${data.num_persons_detected} persons)`);
    }
  } catch (err) {
    photo.status = "error";
    photo.error = err.message;
    showToast(`Error analyzing ${photo.filename}: ${err.message}`, "error");
  } finally {
    hideLoading();
    renderBatchScrollBar();
  }
}

// Delete a photo from the scroll bar
function deleteBatchPhoto(photoId, event) {
  if (event) {
    event.stopPropagation();
  }

  const idx = batchPhotos.findIndex((p) => p.id === photoId);
  if (idx === -1) return;

  const photoToDelete = batchPhotos[idx];
  if (photoToDelete.previewUrl) {
    URL.revokeObjectURL(photoToDelete.previewUrl);
  }

  batchPhotos.splice(idx, 1);

  if (activePhotoId === photoId) {
    if (batchPhotos.length > 0) {
      const nextIdx = Math.min(idx, batchPhotos.length - 1);
      selectBatchPhoto(batchPhotos[nextIdx].id);
    } else {
      activePhotoId = null;
      currentDetectionData = null;
      if (workspaceSection) workspaceSection.classList.add("hidden");
      const container = document.getElementById("batch-gallery-container");
      if (container) container.classList.add("hidden");
    }
  }

  renderBatchScrollBar();
  showToast(`Deleted ${photoToDelete.filename} from batch`);
}

// Clear all photos from current session
function clearAllBatchPhotos() {
  if (batchPhotos.length === 0) return;
  batchPhotos.forEach((p) => {
    if (p.previewUrl) URL.revokeObjectURL(p.previewUrl);
  });
  batchPhotos = [];
  activePhotoId = null;
  currentDetectionData = null;

  if (workspaceSection) workspaceSection.classList.add("hidden");
  const container = document.getElementById("batch-gallery-container");
  if (container) container.classList.add("hidden");

  renderBatchScrollBar();
  showToast("Cleared all photos from session 🗑️");
}

// Compatibility wrapper
async function uploadImageFile(file) {
  await handleIncomingFiles([file]);
}

// Render Results to UI
function renderDetectionResults(data) {
  document.getElementById("active-filename").textContent = data.filename;
  document.getElementById("active-dimensions").textContent = `${data.image_dimensions.width}x${data.image_dimensions.height}px`;

  const numPersons = data.num_persons_detected;
  document.getElementById("active-count").textContent = `${numPersons} Person${numPersons === 1 ? "" : "s"}`;
  document.getElementById("person-count-badge").textContent = `${numPersons} person${numPersons === 1 ? "" : "s"} found`;

  updateDisplayedImage();
  renderPersonCards(data.persons, data.person_crops || {});
  updateJsonBlock(data);
  updatePersonSummaryCounts();
}

// View Mode Switching
function setViewMode(mode) {
  currentViewMode = mode;
  const activeP = getActivePhoto();
  if (activeP) {
    activeP.viewMode = mode;
  }
  const btnAnnotated = document.getElementById("btn-view-annotated");
  const btnOriginal = document.getElementById("btn-view-original");
  const btnCleaned = document.getElementById("btn-view-cleaned");
  const btnRevert = document.getElementById("btn-revert-original");
  const modeName = document.getElementById("view-mode-name");

  [btnAnnotated, btnOriginal, btnCleaned].forEach((btn) => {
    if (!btn) return;
    btn.className = "px-3.5 py-1.5 font-semibold rounded-lg text-zinc-400 hover:text-white transition-all flex items-center gap-1.5";
  });

  if (mode === "annotated" && btnAnnotated) {
    btnAnnotated.className = "px-3.5 py-1.5 font-semibold rounded-lg bg-indigo-600 text-white shadow-sm transition-all flex items-center gap-1.5";
    if (modeName) modeName.textContent = "Annotated Detection";
  } else if (mode === "original" && btnOriginal) {
    btnOriginal.className = "px-3.5 py-1.5 font-semibold rounded-lg bg-indigo-600 text-white shadow-sm transition-all flex items-center gap-1.5";
    if (modeName) modeName.textContent = "Original Untouched";
  } else if (mode === "cleaned" && btnCleaned) {
    btnCleaned.classList.remove("hidden");
    btnCleaned.className = "px-3.5 py-1.5 font-semibold rounded-lg bg-emerald-600 text-white shadow-sm transition-all flex items-center gap-1.5";
    if (modeName) modeName.textContent = "Photobombers Erased";
    if (btnRevert) btnRevert.classList.remove("hidden");
  }

  updateDisplayedImage();
}

function updateDisplayedImage() {
  if (!currentDetectionData) return;

  let src = currentDetectionData.original_image;
  if (currentViewMode === "cleaned" && currentDetectionData.cleaned_image) {
    src = currentDetectionData.cleaned_image;
  }

  imageDisplay.onload = () => {
    resizeInteractiveCanvas();
    drawInteractiveCanvas();
  };

  imageDisplay.src = src;

  if (imageDisplay.complete && imageDisplay.naturalWidth > 0) {
    resizeInteractiveCanvas();
    drawInteractiveCanvas();
  }
}

// Render Person Cards with Micro-Meters & Dedicated Action Buttons
function renderPersonCards(persons, crops) {
  if (!persons || persons.length === 0) {
    personsList.innerHTML = `
      <div class="bg-black/30 border border-white/5 rounded-xl p-5 text-center text-xs text-zinc-500">
        No persons localized in this photograph.
      </div>
    `;
    return;
  }

  const imgW = currentDetectionData ? currentDetectionData.image_dimensions.width : 1;
  const imgH = currentDetectionData ? currentDetectionData.image_dimensions.height : 1;
  const totalImgArea = imgW * imgH;

  personsList.innerHTML = persons
    .map((p) => {
      const activeP = getActivePhoto();
      const isErased = Boolean(p.is_erased || (activeP && activeP.erasedInstanceIds && activeP.erasedInstanceIds.has(p.instance_id)));
      const isKeep = p.classification === "keep" && !isErased;
      const [x1, y1, x2, y2] = p.bounding_box;
      const boxArea = (x2 - x1) * (y2 - y1);
      const isLargeSubject = boxArea / totalImgArea > 0.35;

      const cardBorderCls = isErased ? "border-amber-500/40 bg-amber-500/[0.03]" : isKeep ? "person-card-keep" : "person-card-remove";
      const cropSrc = crops[String(p.instance_id)] || "";
      const sigs = p.signals_used || {};

      return `
      <div id="person-card-${p.instance_id}" 
           onmouseenter="onPersonCardHover(${p.instance_id})" 
           onmouseleave="onPersonCardLeave()" 
           class="glass-panel ${cardBorderCls} rounded-xl p-4 space-y-3 transition-all">
        
        <div class="flex items-center justify-between gap-2">
          <div class="flex items-center gap-3 min-w-0">
            ${
              cropSrc
                ? `<div class="w-11 h-11 rounded-lg overflow-hidden border border-white/10 shadow-md bg-black/60 flex-shrink-0">
                     <img src="${cropSrc}" class="w-full h-full object-cover">
                   </div>`
                : ""
            }
            <div class="min-w-0">
              <div class="flex items-center gap-1.5 flex-wrap">
                <span class="text-xs font-bold text-white">Instance #${p.instance_id}</span>
                ${
                  isErased
                    ? `<span class="px-2 py-0.5 text-[10px] font-bold rounded-full bg-amber-500/15 text-amber-400 border border-amber-500/30 flex items-center gap-1">
                         <span>✨ Erased</span>
                       </span>`
                    : ""
                }
                <span class="text-[10px] font-mono text-zinc-500">[${p.bounding_box.join(", ")}]</span>
              </div>
              <div class="text-[11px] font-mono text-zinc-400 mt-0.5">
                Confidence: <strong class="text-white">${(p.confidence_score * 100).toFixed(1)}%</strong>
              </div>
            </div>
          </div>

          <div class="flex items-center gap-2 flex-shrink-0">
            ${
              isLargeSubject && !isErased
                ? `<span class="hidden sm:inline-block px-2 py-0.5 text-[10px] text-emerald-400 font-semibold bg-emerald-500/10 rounded border border-emerald-500/20" title="Primary foreground subject">
                     🛡️ Main
                   </span>`
                : ""
            }
            
            <!-- Segmented Option Control: [ Keep ] / [ Remove ] -->
            <div class="inline-flex rounded-xl p-0.5 bg-black/60 border border-white/10 shadow-inner">
              <button type="button"
                      onclick="setPersonClassification(${p.instance_id}, 'keep', event)"
                      title="${isErased ? 'Revert & restore this person back into photo' : 'Keep this person as intended subject'}"
                      class="px-2.5 py-1 rounded-lg text-xs font-bold transition-all flex items-center gap-1 cursor-pointer ${
                        isKeep
                          ? 'bg-emerald-600 text-white shadow-md shadow-emerald-500/30'
                          : 'text-zinc-400 hover:text-white'
                      }">
                <span>✓ Keep</span>
              </button>
              <button type="button"
                      onclick="setPersonClassification(${p.instance_id}, 'remove', event)"
                      title="Mark this person in red to be removed from background"
                      class="px-2.5 py-1 rounded-lg text-xs font-bold transition-all flex items-center gap-1 cursor-pointer ${
                        !isKeep
                          ? 'bg-rose-600 text-white shadow-md shadow-rose-500/30'
                          : 'text-zinc-400 hover:text-white'
                      }">
                <span>✕ Remove</span>
              </button>
            </div>

            ${
              isErased
                ? `<button type="button"
                           onclick="revertSinglePerson(${p.instance_id}, event)"
                           title="Restore this person back into the photograph"
                           class="px-2.5 py-1 text-xs font-bold rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white shadow-md shadow-indigo-600/30 flex items-center gap-1 transition-all cursor-pointer">
                     <span>↺ Revert</span>
                   </button>`
                : !isKeep
                ? `<button type="button"
                           onclick="eraseSinglePerson(${p.instance_id}, event)"
                           title="Directly erase this person from background"
                           class="px-2.5 py-1 text-xs font-bold rounded-lg bg-rose-600 hover:bg-rose-500 text-white shadow-md shadow-rose-600/30 flex items-center gap-1 transition-all cursor-pointer">
                     <span>🪄 Erase</span>
                   </button>`
                : ""
            }
          </div>
        </div>

        <!-- 5 Micro-Meters -->
        <div class="space-y-1.5 pt-2 border-t border-white/[0.06]">
          <div class="text-[9px] uppercase font-bold text-zinc-500 tracking-wider">Signals Decomposition</div>
          ${renderMicroMeter("🎯 Centrality", sigs.centrality || 0)}
          ${renderMicroMeter("📐 Relative Size", sigs.relative_size || 0)}
          ${renderMicroMeter("🔍 Focus / Sharpness", sigs.focus || 0)}
          ${renderMicroMeter("🌊 Depth Disparity", sigs.depth || 0)}
          ${renderMicroMeter("🧍 Pose Visibility", sigs.pose || 0)}
        </div>

      </div>
    `;
    })
    .join("");
}

function renderMicroMeter(name, val) {
  const pct = Math.round(val * 100);
  const color = val >= 0.6 ? "bg-emerald-400" : val >= 0.4 ? "bg-amber-400" : "bg-rose-400";
  return `
    <div class="space-y-0.5">
      <div class="flex justify-between text-[11px]">
        <span class="text-zinc-400">${name}</span>
        <span class="font-mono text-zinc-300 font-semibold">${val.toFixed(2)}</span>
      </div>
      <div class="w-full bg-zinc-900/90 h-1.5 rounded-full overflow-hidden border border-white/5">
        <div class="${color} h-full rounded-full transition-all duration-300" style="width: ${pct}%"></div>
      </div>
    </div>
  `;
}

// Canvas Bounding Box Rendering & Hover
function setupCanvasListeners() {
  interactiveCanvas.addEventListener("mousemove", (e) => {
    if (!currentDetectionData || !currentDetectionData.persons) return;

    const rect = interactiveCanvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const scaleX = currentDetectionData.image_dimensions.width / rect.width;
    const scaleY = currentDetectionData.image_dimensions.height / rect.height;

    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    let foundId = null;
    for (const p of currentDetectionData.persons) {
      const [x1, y1, x2, y2] = p.bounding_box;
      if (mouseX >= x1 && mouseX <= x2 && mouseY >= y1 && mouseY <= y2) {
        foundId = p.instance_id;
        break;
      }
    }

    if (foundId !== hoveredPersonId) {
      hoveredPersonId = foundId;
      drawInteractiveCanvas();
      highlightPersonCard(hoveredPersonId);
    }
  });

  interactiveCanvas.addEventListener("mouseleave", () => {
    hoveredPersonId = null;
    drawInteractiveCanvas();
    highlightPersonCard(null);
  });

  interactiveCanvas.addEventListener("click", (e) => {
    if (!currentDetectionData || !currentDetectionData.persons) return;
    if (hoveredPersonId !== null) {
      togglePersonClassification(hoveredPersonId, e);
    }
  });

  if (window.ResizeObserver && imageDisplay) {
    const ro = new ResizeObserver(() => {
      resizeInteractiveCanvas();
      drawInteractiveCanvas();
    });
    ro.observe(imageDisplay);
  }
}

function resizeInteractiveCanvas() {
  if (!imageDisplay || !interactiveCanvas) return;
  const w = imageDisplay.clientWidth || imageDisplay.offsetWidth;
  const h = imageDisplay.clientHeight || imageDisplay.offsetHeight;
  if (w > 0 && h > 0) {
    interactiveCanvas.width = w;
    interactiveCanvas.height = h;
  }
}

window.addEventListener("resize", () => {
  resizeInteractiveCanvas();
  drawInteractiveCanvas();
});

// Modern Viewfinder Bounding Box Rendering
function drawInteractiveCanvas() {
  const ctx = interactiveCanvas.getContext("2d");
  ctx.clearRect(0, 0, interactiveCanvas.width, interactiveCanvas.height);

  if (!currentDetectionData || !currentDetectionData.persons) return;
  if (currentViewMode !== "annotated") return;

  const imgW = currentDetectionData.image_dimensions.width;
  const imgH = currentDetectionData.image_dimensions.height;
  const scaleX = interactiveCanvas.width / imgW;
  const scaleY = interactiveCanvas.height / imgH;

  currentDetectionData.persons.forEach((p) => {
    const [x1, y1, x2, y2] = p.bounding_box;
    const cx1 = x1 * scaleX;
    const cy1 = y1 * scaleY;
    const cw = (x2 - x1) * scaleX;
    const ch = (y2 - y1) * scaleY;

    const isHovered = p.instance_id === hoveredPersonId;
    const isKeep = p.classification === "keep";
    const primaryColor = isKeep ? "#10b981" : "#f43f5e";

    // Glow Effect
    ctx.save();
    if (isHovered) {
      ctx.shadowColor = primaryColor;
      ctx.shadowBlur = 16;
    }

    // Modern Rounded Rectangle Box
    ctx.strokeStyle = primaryColor;
    ctx.lineWidth = isHovered ? 3.5 : 2;

    if (ctx.roundRect) {
      ctx.beginPath();
      ctx.roundRect(cx1, cy1, cw, ch, 6);
      ctx.stroke();
    } else {
      ctx.strokeRect(cx1, cy1, cw, ch);
    }

    // Viewfinder Corner Brackets
    const bracketLen = Math.min(14, cw * 0.25, ch * 0.25);
    ctx.lineWidth = isHovered ? 4 : 3;
    ctx.strokeStyle = "#ffffff";
    
    // Top-left
    ctx.beginPath();
    ctx.moveTo(cx1, cy1 + bracketLen);
    ctx.lineTo(cx1, cy1);
    ctx.lineTo(cx1 + bracketLen, cy1);
    ctx.stroke();

    // Top-right
    ctx.beginPath();
    ctx.moveTo(cx1 + cw - bracketLen, cy1);
    ctx.lineTo(cx1 + cw, cy1);
    ctx.lineTo(cx1 + cw, cy1 + bracketLen);
    ctx.stroke();

    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(cx1, cy1 + ch - bracketLen);
    ctx.lineTo(cx1, cy1 + ch);
    ctx.lineTo(cx1 + bracketLen, cy1 + ch);
    ctx.stroke();

    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(cx1 + cw - bracketLen, cy1 + ch);
    ctx.lineTo(cx1 + cw, cy1 + ch);
    ctx.lineTo(cx1 + cw, cy1 + ch - bracketLen);
    ctx.stroke();

    if (isHovered) {
      ctx.fillStyle = isKeep ? "rgba(16, 185, 129, 0.18)" : "rgba(244, 63, 94, 0.18)";
      ctx.fillRect(cx1, cy1, cw, ch);
    }
    ctx.restore();

    // Glass Badge Label
    const label = `ID #${p.instance_id}: ${p.classification.toUpperCase()} (${(p.confidence_score * 100).toFixed(0)}%)`;
    ctx.font = "bold 11px 'Plus Jakarta Sans', sans-serif";
    const textMetrics = ctx.measureText(label);
    const textW = textMetrics.width + 12;
    const textH = 20;

    const labelY = Math.max(2, cy1 - textH - 3);

    // Pill background
    ctx.fillStyle = primaryColor;
    if (ctx.roundRect) {
      ctx.beginPath();
      ctx.roundRect(cx1, labelY, textW, textH, 4);
      ctx.fill();
    } else {
      ctx.fillRect(cx1, labelY, textW, textH);
    }

    // Pill text
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, cx1 + 6, labelY + 14);
  });
}

function onPersonCardHover(instanceId) {
  hoveredPersonId = instanceId;
  drawInteractiveCanvas();
  highlightPersonCard(instanceId);
}

function onPersonCardLeave() {
  hoveredPersonId = null;
  drawInteractiveCanvas();
  highlightPersonCard(null);
}

function highlightPersonCard(instanceId) {
  document.querySelectorAll("[id^='person-card-']").forEach((el) => {
    el.classList.remove("person-card-highlighted");
  });
  if (instanceId !== null) {
    const card = document.getElementById(`person-card-${instanceId}`);
    if (card) {
      card.classList.add("person-card-highlighted");
      card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }
}

// Dynamic Summary Counts & Action Button State
function updatePersonSummaryCounts() {
  if (!currentDetectionData || !currentDetectionData.persons) return;
  const persons = currentDetectionData.persons;
  const numTotal = persons.length;
  const numRemove = persons.filter((p) => p.classification === "remove").length;
  const numKeep = numTotal - numRemove;

  const activeCountEl = document.getElementById("active-count");
  if (activeCountEl) {
    activeCountEl.innerHTML = `<span>${numTotal} Person${numTotal === 1 ? "" : "s"}</span> &bull; <span class="text-rose-400 font-bold">${numRemove} Remove</span> &bull; <span class="text-emerald-400 font-bold">${numKeep} Keep</span>`;
  }

  const badgeEl = document.getElementById("person-count-badge");
  if (badgeEl) {
    badgeEl.textContent = `${numTotal} persons (${numRemove} marked for removal)`;
  }

  const removeBtn = document.getElementById("btn-inpaint-all");
  const removeBtnText = document.getElementById("btn-remove-text");
  if (removeBtn) {
    if (numRemove > 0) {
      removeBtn.disabled = false;
      removeBtn.classList.remove("opacity-40", "cursor-not-allowed");
      removeBtn.classList.add("cursor-pointer");
      if (removeBtnText) {
        removeBtnText.textContent = `🪄 AI Erase ${numRemove} Person${numRemove > 1 ? "s" : ""} (Keep ${numKeep})`;
      }
    } else {
      removeBtn.disabled = true;
      removeBtn.classList.add("opacity-40", "cursor-not-allowed");
      removeBtn.classList.remove("cursor-pointer");
      if (removeBtnText) {
        removeBtnText.textContent = "Remove Photobombers";
      }
    }
  }
}

// Set Person Classification Directly to 'keep' or 'remove'
function setPersonClassification(instanceId, targetClass, e) {
  if (e) e.stopPropagation();
  if (!currentDetectionData || !currentDetectionData.persons) return;

  const p = currentDetectionData.persons.find((x) => x.instance_id === instanceId);
  if (!p) return;

  const activeP = getActivePhoto();
  const isCurrentlyErased = Boolean(
    p.is_erased || (activeP && activeP.erasedInstanceIds && activeP.erasedInstanceIds.has(instanceId))
  );

  // If person is currently erased and user clicks 'keep', seamlessly restore them!
  if (isCurrentlyErased && targetClass === "keep") {
    revertSinglePerson(instanceId, e);
    return;
  }

  if (p.classification === targetClass && !isCurrentlyErased) return;

  p.classification = targetClass;

  renderPersonCards(currentDetectionData.persons, currentDetectionData.person_crops || {});
  drawInteractiveCanvas();
  updateJsonBlock(currentDetectionData);
  updatePersonSummaryCounts();

  const newBadge = targetClass.toUpperCase();
  const icon = targetClass === "keep" ? "✅" : "❌";
  showToast(`${icon} Instance #${instanceId} set to ${newBadge}`);
}

// Interactive Manual Override Toggle for KEEP / REMOVE
function togglePersonClassification(instanceId, e) {
  if (e) e.stopPropagation();
  if (!currentDetectionData || !currentDetectionData.persons) return;

  const p = currentDetectionData.persons.find((x) => x.instance_id === instanceId);
  if (!p) return;

  const newClass = p.classification === "keep" ? "remove" : "keep";
  setPersonClassification(instanceId, newClass, e);
}

// AI Inpainting & Photobomb Removal Engine (Gemini AI & Silhouette Matching)
async function inpaintPhotobombers() {
  if (!currentDetectionData || !currentDetectionData.original_image) {
    showToast("No photograph loaded to process", "error");
    return;
  }

  const removeBoxes = (currentDetectionData.persons || [])
    .filter((p) => p.classification === "remove")
    .map((p) => p.bounding_box);

  const keepBoxes = (currentDetectionData.persons || [])
    .filter((p) => p.classification === "keep")
    .map((p) => p.bounding_box);

  if (removeBoxes.length === 0) {
    showToast("No photobombers marked in red for removal in this photo!", "info");
    return;
  }

  showLoading(`AI Engine removing ${removeBoxes.length} person(s) & matching background...`);
  try {
    const geminiKey = localStorage.getItem("gemini_api_key") || null;

    const res = await fetch("/api/inpaint", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_b64: currentDetectionData.original_image,
        boxes_to_remove: removeBoxes,
        keep_boxes: keepBoxes,
        gemini_api_key: geminiKey,
        use_ai: true,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Removal failed");
    }

    const data = await res.json();
    currentDetectionData.cleaned_image = data.cleaned_image;
    const activeP = getActivePhoto();
    const removedIds = new Set(
      (currentDetectionData.persons || [])
        .filter((p) => p.classification === "remove")
        .map((p) => p.instance_id)
    );
    if (activeP) {
      activeP.cleanedImage = data.cleaned_image;
      activeP.viewMode = "cleaned";
      activeP.status = "cleaned";
      activeP.erasedInstanceIds = removedIds;
    }
    // Mark erased flag on person instances without deleting them
    (currentDetectionData.persons || []).forEach((p) => {
      if (removedIds.has(p.instance_id)) {
        p.is_erased = true;
      }
    });

    setViewMode("cleaned");
    renderPersonCards(currentDetectionData.persons, currentDetectionData.person_crops || {});
    renderBatchScrollBar();
    updateJsonBlock(currentDetectionData);
    updatePersonSummaryCounts();
    const engineLabel = data.engine === "google-gemini-ai" ? "Gemini Generative AI" : "AI Inpainter";
    showToast(`✨ ${engineLabel} removed ${data.removed_count} person(s) with matching background!`);
  } catch (err) {
    showToast("Error removing photobombers: " + err.message, "error");
  } finally {
    hideLoading();
  }
}

// Erase a Single Targeted Person
async function eraseSinglePerson(instanceId, e) {
  if (e) e.stopPropagation();
  if (!currentDetectionData || !currentDetectionData.original_image) return;

  const p = (currentDetectionData.persons || []).find((x) => x.instance_id === instanceId);
  if (!p) return;

  const activeP = getActivePhoto();
  if (activeP) {
    activeP.erasedInstanceIds = activeP.erasedInstanceIds || new Set();
    activeP.erasedInstanceIds.add(instanceId);
  }
  p.is_erased = true;
  p.classification = "remove";

  showLoading(`AI Engine erasing Instance #${instanceId} with background match...`);
  try {
    const allErasedIds = activeP ? activeP.erasedInstanceIds : new Set([instanceId]);
    const removeBoxes = (currentDetectionData.persons || [])
      .filter((x) => allErasedIds.has(x.instance_id))
      .map((x) => x.bounding_box);

    const keepBoxes = (currentDetectionData.persons || [])
      .filter((x) => !allErasedIds.has(x.instance_id))
      .map((x) => x.bounding_box);

    const geminiKey = localStorage.getItem("gemini_api_key") || null;

    const res = await fetch("/api/inpaint", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_b64: currentDetectionData.original_image,
        boxes_to_remove: removeBoxes,
        keep_boxes: keepBoxes,
        gemini_api_key: geminiKey,
        use_ai: true,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Inpainting failed");
    }

    const data = await res.json();
    currentDetectionData.cleaned_image = data.cleaned_image;
    if (activeP) {
      activeP.cleanedImage = data.cleaned_image;
      activeP.viewMode = "cleaned";
      activeP.status = "cleaned";
    }
    setViewMode("cleaned");
    renderPersonCards(currentDetectionData.persons, currentDetectionData.person_crops || {});
    renderBatchScrollBar();
    updateJsonBlock(currentDetectionData);
    updatePersonSummaryCounts();
    showToast(`✨ Erased Instance #${instanceId} with matching background!`);
  } catch (err) {
    showToast("Error erasing person: " + err.message, "error");
  } finally {
    hideLoading();
  }
}

// Revert / Restore a Single Erased Person
async function revertSinglePerson(instanceId, e) {
  if (e) e.stopPropagation();
  if (!currentDetectionData || !currentDetectionData.original_image) return;

  const p = (currentDetectionData.persons || []).find((x) => x.instance_id === instanceId);
  if (!p) return;

  p.is_erased = false;
  p.classification = "keep";

  const activeP = getActivePhoto();
  if (activeP && activeP.erasedInstanceIds) {
    activeP.erasedInstanceIds.delete(instanceId);
  }

  const remainingErased = (currentDetectionData.persons || []).filter(
    (x) => x.is_erased || (activeP && activeP.erasedInstanceIds && activeP.erasedInstanceIds.has(x.instance_id))
  );

  if (remainingErased.length === 0) {
    currentDetectionData.cleaned_image = null;
    if (activeP) {
      activeP.cleanedImage = null;
      activeP.status = "ready";
      activeP.viewMode = "annotated";
      if (activeP.erasedInstanceIds) activeP.erasedInstanceIds.clear();
    }
    setViewMode("annotated");
    const btnCleaned = document.getElementById("btn-view-cleaned");
    const btnRevert = document.getElementById("btn-revert-original");
    if (btnCleaned) btnCleaned.classList.add("hidden");
    if (btnRevert) btnRevert.classList.add("hidden");
    renderPersonCards(currentDetectionData.persons, currentDetectionData.person_crops || {});
    drawInteractiveCanvas();
    updateJsonBlock(currentDetectionData);
    updatePersonSummaryCounts();
    renderBatchScrollBar();
    showToast(`↺ Restored Instance #${instanceId} back to photo!`);
    return;
  }

  showLoading(`Restoring Instance #${instanceId}...`);
  try {
    const removeBoxes = remainingErased.map((x) => x.bounding_box);
    const keepBoxes = (currentDetectionData.persons || [])
      .filter((x) => !remainingErased.some((r) => r.instance_id === x.instance_id))
      .map((x) => x.bounding_box);

    const geminiKey = localStorage.getItem("gemini_api_key") || null;
    const res = await fetch("/api/inpaint", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_b64: currentDetectionData.original_image,
        boxes_to_remove: removeBoxes,
        keep_boxes: keepBoxes,
        gemini_api_key: geminiKey,
        use_ai: true,
      }),
    });

    if (!res.ok) {
      throw new Error("Failed to restore person");
    }

    const data = await res.json();
    currentDetectionData.cleaned_image = data.cleaned_image;
    if (activeP) {
      activeP.cleanedImage = data.cleaned_image;
      activeP.viewMode = "cleaned";
    }
    setViewMode("cleaned");
    renderPersonCards(currentDetectionData.persons, currentDetectionData.person_crops || {});
    drawInteractiveCanvas();
    updateJsonBlock(currentDetectionData);
    updatePersonSummaryCounts();
    renderBatchScrollBar();
    showToast(`↺ Restored Instance #${instanceId} back to photo!`);
  } catch (err) {
    showToast("Error restoring person: " + err.message, "error");
  } finally {
    hideLoading();
  }
}

// 1-Click Instant Restore to Original Image
function revertToOriginal() {
  if (!currentDetectionData) return;
  currentDetectionData.cleaned_image = null;
  (currentDetectionData.persons || []).forEach((p) => {
    p.is_erased = false;
  });
  const activeP = getActivePhoto();
  if (activeP) {
    activeP.cleanedImage = null;
    activeP.viewMode = "annotated";
    activeP.status = "ready";
    if (activeP.erasedInstanceIds) activeP.erasedInstanceIds.clear();
  }
  const btnCleaned = document.getElementById("btn-view-cleaned");
  const btnRevert = document.getElementById("btn-revert-original");
  if (btnCleaned) btnCleaned.classList.add("hidden");
  if (btnRevert) btnRevert.classList.add("hidden");
  setViewMode("annotated");
  renderPersonCards(currentDetectionData.persons, currentDetectionData.person_crops || {});
  renderBatchScrollBar();
  drawInteractiveCanvas();
  updateJsonBlock(currentDetectionData);
  updatePersonSummaryCounts();
  showToast("Restored original photograph ↺");
}

// Google Gemini Settings Modal Handlers
function openGeminiModal() {
  const modal = document.getElementById("gemini-modal");
  const input = document.getElementById("gemini-api-key-input");
  if (input) input.value = localStorage.getItem("gemini_api_key") || "";
  if (modal) modal.classList.remove("hidden");
}

function closeGeminiModal() {
  const modal = document.getElementById("gemini-modal");
  if (modal) modal.classList.add("hidden");
}

function saveGeminiApiKey() {
  const input = document.getElementById("gemini-api-key-input");
  if (input) {
    const key = input.value.trim();
    if (key) {
      localStorage.setItem("gemini_api_key", key);
      showToast("Gemini API Key saved! ✨ Generative inpainting enabled.");
    } else {
      localStorage.removeItem("gemini_api_key");
      showToast("Reset to built-in AI background engine");
    }
  }
  closeGeminiModal();
}

// Live Parameter Tuning (Sub-Millisecond Reclassify)
let reclassifyDebounce = null;

function onThresholdChange(val) {
  document.getElementById("threshold-val").textContent = parseFloat(val).toFixed(2);
  const activeP = getActivePhoto();
  if (activeP) {
    activeP.threshold = parseFloat(val);
  }
  triggerLiveReclassification();
}

function onWeightsChange() {
  const wCent = parseInt(document.getElementById("w-centrality").value);
  const wSize = parseInt(document.getElementById("w-size").value);
  const wFoc = parseInt(document.getElementById("w-focus").value);
  const wDep = parseInt(document.getElementById("w-depth").value);
  const wPos = parseInt(document.getElementById("w-pose").value);

  const total = wCent + wSize + wFoc + wDep + wPos || 1;

  document.getElementById("w-centrality-val").textContent = `${Math.round((wCent / total) * 100)}%`;
  document.getElementById("w-size-val").textContent = `${Math.round((wSize / total) * 100)}%`;
  document.getElementById("w-focus-val").textContent = `${Math.round((wFoc / total) * 100)}%`;
  document.getElementById("w-depth-val").textContent = `${Math.round((wDep / total) * 100)}%`;
  document.getElementById("w-pose-val").textContent = `${Math.round((wPos / total) * 100)}%`;

  triggerLiveReclassification();
}

function getNormalizedWeights() {
  const wCent = parseFloat(document.getElementById("w-centrality").value);
  const wSize = parseFloat(document.getElementById("w-size").value);
  const wFoc = parseFloat(document.getElementById("w-focus").value);
  const wDep = parseFloat(document.getElementById("w-depth").value);
  const wPos = parseFloat(document.getElementById("w-pose").value);
  const total = wCent + wSize + wFoc + wDep + wPos || 1.0;

  return {
    centrality: wCent / total,
    relative_size: wSize / total,
    focus: wFoc / total,
    depth: wDep / total,
    pose: wPos / total,
  };
}

function triggerLiveReclassification() {
  if (!currentDetectionData || !currentDetectionData.persons) return;

  clearTimeout(reclassifyDebounce);
  reclassifyDebounce = setTimeout(async () => {
    const threshold = parseFloat(document.getElementById("threshold-slider").value);
    const weights = getNormalizedWeights();

    try {
      const res = await fetch("/api/reclassify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          persons: currentDetectionData.persons,
          threshold: threshold,
          weights: weights,
        }),
      });

      if (!res.ok) return;
      const data = await res.json();

      currentDetectionData.persons = data.persons;
      const activeP = getActivePhoto();
      if (activeP && activeP.detectionData) {
        activeP.detectionData.persons = data.persons;
      }
      renderPersonCards(currentDetectionData.persons, currentDetectionData.person_crops || {});
      drawInteractiveCanvas();
      updateJsonBlock(currentDetectionData);
      updatePersonSummaryCounts();
    } catch (e) {
      console.error("Reclassify error:", e);
    }
  }, 35);
}

function resetTunerDefaults() {
  document.getElementById("threshold-slider").value = "0.62";
  document.getElementById("threshold-val").textContent = "0.62";

  document.getElementById("w-centrality").value = DEFAULT_WEIGHTS.centrality;
  document.getElementById("w-size").value = DEFAULT_WEIGHTS.size;
  document.getElementById("w-focus").value = DEFAULT_WEIGHTS.focus;
  document.getElementById("w-depth").value = DEFAULT_WEIGHTS.depth;
  document.getElementById("w-pose").value = DEFAULT_WEIGHTS.pose;

  onWeightsChange();
  showToast("Reset sliders to calibrated defaults");
}

function toggleWeightsPanel() {
  const panel = document.getElementById("weights-panel");
  const icon = document.getElementById("weights-toggle-icon");
  if (panel.classList.contains("hidden")) {
    panel.classList.remove("hidden");
    icon.textContent = "▲";
  } else {
    panel.classList.add("hidden");
    icon.textContent = "▼";
  }
}

// Bottom Studio Tab Switching
function switchBottomTab(tabId) {
  ["json", "benchmark", "architecture"].forEach((t) => {
    document.getElementById(`tab-pane-${t}`).classList.add("hidden");
    const btn = document.getElementById(`tab-btn-${t}`);
    btn.className = "px-5 py-3.5 text-xs font-bold border-b-2 border-transparent text-zinc-400 hover:text-white flex items-center gap-2 whitespace-nowrap transition-all";
  });

  document.getElementById(`tab-pane-${tabId}`).classList.remove("hidden");
  document.getElementById(`tab-btn-${tabId}`).className = "px-5 py-3.5 text-xs font-bold border-b-2 border-indigo-500 text-indigo-400 flex items-center gap-2 whitespace-nowrap transition-all";
}

// JSON Block Formatting & Syntax Simulation
function updateJsonBlock(data) {
  const formatted = {
    image: data.filename,
    dimensions: data.image_dimensions,
    num_persons_detected: data.num_persons_detected,
    persons: data.persons.map((p) => ({
      instance_id: p.instance_id,
      bounding_box: p.bounding_box,
      classification: p.classification,
      confidence_score: p.confidence_score,
      signals_used: p.signals_used,
    })),
  };
  jsonPreBlock.textContent = JSON.stringify(formatted, null, 2);
}

function copyJsonToClipboard() {
  const text = jsonPreBlock.textContent;
  navigator.clipboard.writeText(text);
  showToast("JSON copied to clipboard! 📋");
}

function downloadJsonFile() {
  const text = jsonPreBlock.textContent;
  const blob = new Blob([text], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${currentDetectionData ? currentDetectionData.filename.replace(/\.[^/.]+$/, "") : "result"}_spec.json`;
  a.click();
  URL.revokeObjectURL(url);
  showToast("Downloaded structured JSON file 💾");
}

function downloadAnnotatedImage() {
  if (!currentDetectionData || !currentDetectionData.annotated_image) return;
  const a = document.createElement("a");
  a.href = currentDetectionData.annotated_image;
  a.download = `${currentDetectionData.filename.replace(/\.[^/.]+$/, "")}_annotated.jpg`;
  a.click();
  showToast("Downloaded annotated image 🖼️");
}

// Evaluation Benchmark Runner
async function runBenchmarkEvaluation() {
  const btn = document.getElementById("btn-run-eval");
  btn.innerHTML = `<span>⏳ Evaluating Ground Truth...</span>`;
  btn.disabled = true;

  try {
    const res = await fetch("/api/evaluate", { method: "POST" });
    const data = await res.json();
    renderEvaluationResults(data);
    showToast("Benchmark evaluation completed!");
  } catch (err) {
    showToast("Evaluation error: " + err.message, "error");
  } finally {
    btn.innerHTML = `<span>⚡ Run Evaluation Benchmark</span>`;
    btn.disabled = false;
  }
}

async function fetchEvaluationBenchmarkOnLoad() {
  try {
    const res = await fetch("/api/evaluate", { method: "POST" });
    const data = await res.json();
    renderEvaluationResults(data);
  } catch (e) {
    console.error("Evaluation load failed:", e);
  }
}

function renderEvaluationResults(data) {
  if (!data || !data.overall) return;

  const ov = data.overall;
  document.getElementById("metric-f1").textContent = ov.f1.toFixed(4);
  document.getElementById("metric-recall").textContent = `${(ov.recall * 100).toFixed(2)}%`;
  document.getElementById("metric-prec").textContent = `${(ov.precision * 100).toFixed(2)}%`;
  document.getElementById("metric-acc").textContent = `${(ov.accuracy * 100).toFixed(2)}%`;

  const tbody = document.getElementById("eval-table-body");
  const cats = [
    { key: "photobomber", name: "Clear Photobombers", cls: "text-emerald-400" },
    { key: "ambiguous", name: "Ambiguous / Crowded", cls: "text-amber-400" },
    { key: "clean", name: "Clean Negatives", cls: "text-sky-400" },
  ];

  let rowsHtml = "";
  cats.forEach((c) => {
    const m = data.by_category[c.key] || { images: 5, tp: 0, fp: 0, fn: 0, tn: 0, precision: 0, recall: 0, f1: 0, accuracy: 0 };
    rowsHtml += `
      <tr class="hover:bg-white/[0.02] transition-colors">
        <td class="px-4 py-2.5 font-bold ${c.cls}">${c.name}</td>
        <td class="px-3 py-2.5 text-center text-zinc-400">${m.images}</td>
        <td class="px-3 py-2.5 text-center text-emerald-400 font-semibold">${m.tp}</td>
        <td class="px-3 py-2.5 text-center text-rose-400 font-semibold">${m.fp}</td>
        <td class="px-3 py-2.5 text-center text-amber-400 font-semibold">${m.fn}</td>
        <td class="px-3 py-2.5 text-center text-zinc-400">${m.tn}</td>
        <td class="px-4 py-2.5 text-right font-bold text-indigo-400">${m.precision.toFixed(4)}</td>
        <td class="px-4 py-2.5 text-right font-bold text-sky-400">${m.recall.toFixed(4)}</td>
        <td class="px-4 py-2.5 text-right font-bold text-emerald-400">${m.f1.toFixed(4)}</td>
        <td class="px-4 py-2.5 text-right font-bold text-zinc-200">${m.accuracy.toFixed(4)}</td>
      </tr>
    `;
  });

  rowsHtml += `
    <tr class="bg-zinc-900/80 font-bold border-t border-white/10">
      <td class="px-4 py-3 text-indigo-400 font-extrabold">OVERALL SUMMARY</td>
      <td class="px-3 py-3 text-center text-zinc-300">15</td>
      <td class="px-3 py-3 text-center text-emerald-400 font-extrabold">${ov.tp}</td>
      <td class="px-3 py-3 text-center text-rose-400 font-extrabold">${ov.fp}</td>
      <td class="px-3 py-3 text-center text-amber-400 font-extrabold">${ov.fn}</td>
      <td class="px-3 py-3 text-center text-zinc-300 font-extrabold">${ov.tn}</td>
      <td class="px-4 py-3 text-right text-indigo-400 font-extrabold">${ov.precision.toFixed(4)}</td>
      <td class="px-4 py-3 text-right text-sky-400 font-extrabold">${ov.recall.toFixed(4)}</td>
      <td class="px-4 py-3 text-right text-emerald-400 font-extrabold">${ov.f1.toFixed(4)}</td>
      <td class="px-4 py-3 text-right text-white font-extrabold">${ov.accuracy.toFixed(4)}</td>
    </tr>
  `;

  tbody.innerHTML = rowsHtml;
}

// UI Feedback & Toast Notifications
function showToast(message, type = "success") {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  const bg = type === "error" ? "bg-rose-950/90 border-rose-500/40 text-rose-200" : "bg-zinc-900/90 border-white/10 text-zinc-100";
  
  toast.className = `toast-animate px-4 py-2.5 rounded-full border ${bg} backdrop-blur-xl shadow-2xl text-xs font-semibold flex items-center gap-2 pointer-events-auto`;
  toast.innerHTML = `<span>${type === "error" ? "⚠️" : "✨"}</span><span>${message}</span>`;
  
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translate(-50%, 10px) scale(0.95)";
    setTimeout(() => toast.remove(), 300);
  }, 2800);
}

function showLoading(text) {
  loadingText.textContent = text || "Processing...";
  loadingOverlay.classList.remove("hidden");
}

function hideLoading() {
  loadingOverlay.classList.add("hidden");
}
