const $ = (sel) => document.querySelector(sel);
const analyzeBtn = $("#analyzeBtn");
const loader = $("#loader");
const resultBox = $("#result");
const resLang = $("#resLang");
const resLabel = $("#resLabel");
const resScore = $("#resScore");
const langSelect = $("#lang");
const textArea = $("#text");

function show(el){ el.classList.remove("hidden"); }
function hide(el){ el.classList.add("hidden"); }
function percent(x){ return (x*100).toFixed(1) + "%"; }

function renderBars(lang, scores){
  const keys = lang === "vi" ? ["NEGATIVE","NEUTRAL","POSITIVE"] : ["NEGATIVE","POSITIVE"];
  keys.forEach(k=>{
    const wrap = document.createElement("div");
    wrap.className = "bar";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = "0%";
    const label = document.createElement("div");
    label.className = "bar-label";
    const sc = scores[k] ?? 0;
    label.innerHTML = `<span>${k}</span><span>${percent(sc)}</span>`;
    wrap.appendChild(fill);
    setTimeout(()=>{ fill.style.width = percent(sc); }, 30);
  });
}

langSelect.addEventListener("change", (e) => {
  const lang = e.target.value;
  console.clear();
  console.log("Ngôn ngữ được chọn:", lang);
});

textArea.addEventListener("input", () => {
  console.log("Nội dung hiện tại:", textArea.value.trim());
});

analyzeBtn.addEventListener("click", async ()=>{
  const text = textArea.value.trim();
  const lang = langSelect.value;
  console.log("▶️ Bắt đầu phân tích với:", { text, lang });

  if (!text) {
    alert("Nhập nội dung trước.");
    return;
  }

  hide(resultBox);
  show(loader);

  try{
    const r = await fetch("/analyze", {
      method:"POST",
      headers:{ "Content-Type":"application/json" },
      body: JSON.stringify({ text, lang })
    });
    const data = await r.json();
    hide(loader);

    if (!r.ok || data.ok !== true){
      alert(data.error || "Có lỗi xảy ra.");
      console.error("Lỗi phản hồi:", data);
      return;
    }

    const { input, result } = data;
    console.log("Kết quả nhận được:", result);

    resLang.textContent = input.lang.toUpperCase();
    resLabel.textContent = result.label;
    resLabel.style.borderColor =
      result.label === "POSITIVE" ? "#23d2ac" :
      result.label === "NEUTRAL"  ? "#ffd166" : "#ff6b6b";
    resScore.textContent = percent(result.score);

    renderBars(input.lang, result.scores);
    show(resultBox);

  }catch(e){
    hide(loader);
    alert("Lỗi mạng hoặc server.");
    console.error("Lỗi khi gửi request:", e);
  }
});
