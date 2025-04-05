document.addEventListener("DOMContentLoaded", function () {
  const fileInput = document.getElementById("file-input");
  const uploadArea = document.getElementById("upload-area");
  const fileList = document.getElementById("file-list");
  const uploadBtn = document.getElementById("upload-btn");
  const uploadLoader = document.getElementById("upload-loader");
  const uploadStatus = document.getElementById("upload-status");
  const summaryContent = document.getElementById("summary-content");
  const questionInput = document.getElementById("question-input");
  const askBtn = document.getElementById("ask-btn");
  const messages = document.getElementById("messages");
  const chatLoader = document.getElementById("chat-loader");

  let files = [];

  // Handle drag and drop events
  uploadArea.addEventListener("dragover", function (e) {
    e.preventDefault();
    uploadArea.style.borderColor = "#4fc3dc";
    uploadArea.style.backgroundColor = "rgba(79, 195, 220, 0.1)";
  });

  uploadArea.addEventListener("dragleave", function () {
    uploadArea.style.borderColor = "#4fc3dc";
    uploadArea.style.backgroundColor = "";
  });

  uploadArea.addEventListener("drop", function (e) {
    e.preventDefault();
    uploadArea.style.borderColor = "#4fc3dc";
    uploadArea.style.backgroundColor = "";

    handleFiles(e.dataTransfer.files);
  });

  uploadArea.addEventListener("click", function () {
    fileInput.click();
  });

  fileInput.addEventListener("change", function () {
    handleFiles(fileInput.files);
  });

  function handleFiles(selectedFiles) {
    for (let i = 0; i < selectedFiles.length; i++) {
      const file = selectedFiles[i];
      if (file.type === "application/pdf") {
        files.push(file);
      }
    }

    updateFileList();
  }

  function updateFileList() {
    fileList.innerHTML = "";

    files.forEach((file, index) => {
      const li = document.createElement("li");
      li.innerHTML = `
                <span>${file.name} (${formatFileSize(file.size)})</span>
                <button class="remove-btn" data-index="${index}">
                    <i class="fas fa-times"></i>
                </button>
            `;
      fileList.appendChild(li);
    });

    uploadBtn.disabled = files.length === 0;

    // Add event listeners to remove buttons
    document.querySelectorAll(".remove-btn").forEach((btn) => {
      btn.addEventListener("click", function () {
        const index = parseInt(this.getAttribute("data-index"));
        files.splice(index, 1);
        updateFileList();
      });
    });
  }

  function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + " bytes";
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    else return (bytes / 1048576).toFixed(1) + " MB";
  }

  uploadBtn.addEventListener("click", function () {
    if (files.length === 0) return;

    const formData = new FormData();
    files.forEach((file) => {
      formData.append("files[]", file);
    });

    uploadLoader.style.display = "block";
    uploadBtn.disabled = true;

    fetch("/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        uploadLoader.style.display = "none";

        if (data.success) {
          uploadStatus.textContent = data.message;
          uploadStatus.className = "status success";
          uploadStatus.style.display = "block";

          summaryContent.innerHTML = marked.parse(data.summary);

          // Enable chat functionality
          questionInput.disabled = false;
          askBtn.disabled = false;

          // Add a system message
          addMessage(
            "Documents processed! I'm ready to answer your questions.",
            "bot"
          );

          // Clear file list
          files = [];
          updateFileList();
        } else {
          uploadStatus.textContent = data.error || "An error occurred";
          uploadStatus.className = "status error";
          uploadStatus.style.display = "block";
          uploadBtn.disabled = false;
        }
      })
      .catch((error) => {
        uploadLoader.style.display = "none";
        uploadStatus.textContent = "Error: " + error.message;
        uploadStatus.className = "status error";
        uploadStatus.style.display = "block";
        uploadBtn.disabled = files.length > 0;
      });
  });

  // Handle asking questions
  askBtn.addEventListener("click", askQuestion);
  questionInput.addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      askQuestion();
    }
  });

  function askQuestion() {
    const question = questionInput.value.trim();
    if (question === "") return;

    addMessage(question, "user");
    questionInput.value = "";

    chatLoader.style.display = "block";
    questionInput.disabled = true;
    askBtn.disabled = true;

    fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question: question }),
    })
      .then((response) => response.json())
      .then((data) => {
        chatLoader.style.display = "none";
        questionInput.disabled = false;
        askBtn.disabled = false;

        if (data.answer) {
          addMessage(data.answer, "bot");
        } else {
          addMessage(
            "I encountered an error: " + (data.error || "Unknown error"),
            "bot"
          );
        }

        questionInput.focus();
      })
      .catch((error) => {
        chatLoader.style.display = "none";
        questionInput.disabled = false;
        askBtn.disabled = false;
        addMessage("Error: " + error.message, "bot");
      });
  }

  function addMessage(text, sender) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}-message`;
    messageDiv.innerHTML = marked.parse(text);
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
  }
});
