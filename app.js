const form = document.querySelector("#prediction-form");
const result = document.querySelector("#prediction-result");
const errorBox = document.querySelector("#form-error");
const menuButton = document.querySelector(".menu-button");
const nav = document.querySelector(".nav-shell");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

document.documentElement.classList.add("motion-ready");

const revealElements = document.querySelectorAll("[data-reveal]");
revealElements.forEach((element) => {
  const delay = Number(element.dataset.revealDelay || 0);
  element.style.setProperty("--reveal-delay", `${delay}ms`);
});

if (reducedMotion.matches || !("IntersectionObserver" in window)) {
  revealElements.forEach((element) => element.classList.add("is-visible"));
} else {
  const revealObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("is-visible");
      observer.unobserve(entry.target);
    });
  }, { rootMargin: "0px 0px -12%", threshold: 0.12 });

  revealElements.forEach((element) => revealObserver.observe(element));
}

menuButton?.addEventListener("click", () => {
  const open = nav.classList.toggle("menu-open");
  menuButton.setAttribute("aria-expanded", String(open));
});

document.querySelectorAll(".nav-links a").forEach((link) => {
  link.addEventListener("click", () => {
    nav.classList.remove("menu-open");
    menuButton?.setAttribute("aria-expanded", "false");
  });
});

function collectInput(formData) {
  const date = new Date(`${formData.get("date")}T00:00:00`);
  return {
    date: formData.get("date"),
    hour: Number(formData.get("hour")),
    weekday: date.toLocaleDateString("en-US", { weekday: "long" }),
    month: date.getMonth() + 1,
    temperature: Number(formData.get("temperature")),
    humidity: Number(formData.get("humidity")),
    rainfall: Number(formData.get("rainfall")),
    snowfall: Number(formData.get("snowfall")),
    holiday: formData.get("holiday"),
    functioning_day: formData.get("functioning_day"),
  };
}

function validate(data) {
  if (!data.date || Number.isNaN(new Date(`${data.date}T00:00:00`).getTime())) return "请选择有效日期。";
  if (!Number.isInteger(data.hour) || data.hour < 0 || data.hour > 23) return "小时必须在 0–23 之间。";
  if (!Number.isFinite(data.temperature) || data.temperature < -25 || data.temperature > 45) return "温度应在 -25°C 到 45°C 之间。";
  if (!Number.isFinite(data.humidity) || data.humidity < 0 || data.humidity > 100) return "湿度应在 0% 到 100% 之间。";
  if (!Number.isFinite(data.rainfall) || data.rainfall < 0 || data.rainfall > 50) return "降雨量应在 0–50 mm 之间。";
  if (!Number.isFinite(data.snowfall) || data.snowfall < 0 || data.snowfall > 10) return "降雪量应在 0–10 cm 之间。";
  return "";
}

function demoEstimate(data) {
  if (data.functioning_day === "No") return 0;
  const commute = [7, 8, 9, 17, 18, 19].includes(data.hour) ? 410 : 80;
  const daylight = data.hour >= 6 && data.hour <= 22 ? 135 : -60;
  const temperature = Math.max(-120, 18 * data.temperature - 0.38 * data.temperature ** 2);
  const weatherPenalty = data.rainfall * 105 + data.snowfall * 135 + Math.max(0, data.humidity - 65) * 3.5;
  const holidayPenalty = data.holiday === "Holiday" ? 90 : 0;
  return Math.max(0, Math.round(210 + commute + daylight + temperature - weatherPenalty - holidayPenalty));
}

async function requestPrediction(data) {
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!response.ok) throw new Error("Backend unavailable");
    const payload = await response.json();
    if (!Number.isFinite(Number(payload.prediction))) throw new Error("Invalid prediction");
    return { value: Math.max(0, Math.round(Number(payload.prediction))), demo: false };
  } catch {
    return { value: demoEstimate(data), demo: true };
  }
}

function showLoading() {
  result.innerHTML = `
    <div class="result-empty">
      <div class="result-icon" aria-hidden="true">···</div>
      <span>正在计算</span>
      <h3>模型正在读取<br>这一小时的条件。</h3>
    </div>`;
}

function animateNumber(element, target) {
  if (reducedMotion.matches) {
    element.textContent = target.toLocaleString("zh-CN");
    return;
  }

  const duration = 850;
  const startTime = performance.now();
  const easeOut = (progress) => 1 - (1 - progress) ** 3;

  function update(now) {
    const progress = Math.min((now - startTime) / duration, 1);
    element.textContent = Math.round(target * easeOut(progress)).toLocaleString("zh-CN");
    if (progress < 1) requestAnimationFrame(update);
  }

  requestAnimationFrame(update);
}

function showResult(value, isDemo) {
  result.innerHTML = `
    <div class="result-empty">
      <div class="result-badge">${isDemo ? "前端演示估计" : "后端模型已完成推断"}</div>
      <span>预计租借需求</span>
      <div><strong class="result-value">0</strong><span class="result-unit">辆 / 小时</span></div>
      <p class="result-note">${isDemo ? "当前为静态页面演示值；接入 Flask /predict 后将自动显示真实模型结果。" : "结果来自服务启动时加载的已保存 Pipeline。"}</p>
    </div>`;
  animateNumber(result.querySelector(".result-value"), value);
}

form?.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.textContent = "";

  const data = collectInput(new FormData(form));
  const validationError = validate(data);
  if (validationError) {
    errorBox.textContent = validationError;
    return;
  }

  showLoading();
  const prediction = await requestPrediction(data);
  showResult(prediction.value, prediction.demo);
});
