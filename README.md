<script src="{{ bower directory }}/bower-webfontloader/webfont.js" />
<script src="{{ bower directory }}/snap.svg/dist/snap.svg-min.js" />
<script src="{{ bower directory }}/underscore/underscore-min.js" />
<script src="{{ bower directory }}/js-sequence-diagrams/dist/sequence-diagram-min.js" />

Hello World

<div id="diagram">Diagram will be placed here</div>
<script> 
  var d = Diagram.parse("A->B: Does something");
  var options = {theme: 'simple'};
  d.drawSVG('diagram', options);
</script>
