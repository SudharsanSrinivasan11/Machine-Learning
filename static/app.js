(() => {
  const messages = document.getElementById("messages");
  const form = document.getElementById("composer");
  const input = document.getElementById("text");
  const sendBtn = document.getElementById("send");
  const suggestions = document.getElementById("suggestions");

  const EMOJI = {
    happy: "😊", sad: "😢", angry: "😠", surprised: "😮", fearful: "😨",
    disgusted: "🤢", confused: "😕", bored: "🥱", excited: "🤩", anxious: "😰",
    ashamed: "😳", jealous: "😒", guilty: "😔", proud: "😌", hopeful: "🌱",
    lonely: "🫂", disappointed: "😞", grateful: "🙏", curious: "🤔",
    relieved: "😮‍💨", loving: "❤️", neutral: "💬", crisis: "🤍",
  };

  const timeNow = () =>
    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  // Messages are inserted with textContent, never innerHTML, so user input cannot inject markup.
  function addMessage(text, who, meta) {
    const row = document.createElement("div");
    row.className = `row ${who}`;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    row.appendChild(bubble);

    const info = document.createElement("div");
    info.className = "meta";
    if (meta && meta.emotion) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = `${EMOJI[meta.emotion] || "💬"} ${meta.emotion}`;
      tag.title = `Detected via ${meta.method} · confidence ${Math.round(meta.confidence * 100)}%`;
      info.appendChild(tag);
    }
    info.appendChild(document.createTextNode(timeNow()));
    row.appendChild(info);

    messages.appendChild(row);
    messages.scrollTop = messages.scrollHeight;
    return row;
  }

  function showTyping() {
    const row = document.createElement("div");
    row.className = "row bot typing";
    row.innerHTML = '<div class="bubble"><span></span><span></span><span></span></div>';
    messages.appendChild(row);
    messages.scrollTop = messages.scrollHeight;
    return row;
  }

  async function send(text) {
    text = text.trim();
    if (!text) return;
    suggestions.hidden = true;
    addMessage(text, "user");
    input.value = "";
    sendBtn.disabled = true;
    const typing = showTyping();

    try {
      const res = await fetch("/get", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ msg: text }),
      });
      const data = await res.json();
      typing.remove();
      if (!res.ok) addMessage(data.error || "Something went wrong.", "bot error");
      else addMessage(data.response, "bot", data);
    } catch (err) {
      typing.remove();
      addMessage("I couldn't reach the server. Is it still running?", "bot error");
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  form.addEventListener("submit", (e) => { e.preventDefault(); send(input.value); });
  suggestions.addEventListener("click", (e) => {
    if (e.target.tagName === "BUTTON") send(e.target.textContent);
  });
  document.getElementById("clear").addEventListener("click", () => {
    messages.replaceChildren();
    suggestions.hidden = false;
    greet();
  });

  const greet = () => addMessage("Hi! I'm here to listen. How are you feeling today?", "bot");
  greet();
  input.focus();
})();
