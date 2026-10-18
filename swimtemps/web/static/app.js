// Forms marked data-confirm ask before submitting. The question is plain text
// from an attribute, never script, so a water's name cannot become code.
document.addEventListener("submit", function (event) {
  var form = event.target.closest("form[data-confirm]");
  if (form && !window.confirm(form.dataset.confirm)) event.preventDefault();
});

// Buttons marked data-copy copy the text of the element whose id they name.
document.addEventListener("click", function (event) {
  var button = event.target.closest("button[data-copy]");
  if (!button) return;
  var source = document.getElementById(button.dataset.copy);
  if (!source || !navigator.clipboard) return;
  navigator.clipboard.writeText(source.textContent).then(function () {
    var label = button.textContent;
    button.textContent = "Copied";
    setTimeout(function () { button.textContent = label; }, 1500);
  });
});
