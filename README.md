<script src="webfont.js"\>
</script\>
<script src="snap.svg-min.js"\></script\>
<script src="underscore-min.js"\></script\>
<script src="sequence-diagram-min.js"\></script\>

Hello World

<div id="diagram"\></div\>
<script\>
  var diagram = Diagram.parse("A->B: Message");
  diagram.drawSVG("diagram", {theme: 'hand'});
</script\>
