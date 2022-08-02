window.MathJax = {
    tex: {
      inlineMath: [["$", "$"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      processHtmlClass: "arithmatex|doc-function"
    }
  };

//   document$.subscribe(() => {
//     MathJax.typesetPromise()
//   })