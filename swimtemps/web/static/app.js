// Forms marked data-confirm ask before submitting. The question is plain text
// from an attribute, never script, so a water's name cannot become code.
document.addEventListener("submit", function (event) {
  var form = event.target.closest("form[data-confirm]");
  if (form && !window.confirm(form.dataset.confirm)) event.preventDefault();
});
