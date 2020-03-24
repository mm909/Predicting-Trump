t = setTimeout(predict, 500);
clearTimeout(t);

function resetTimer() {
  clearTimeout(t);
  t = setTimeout(predict, 500); // time is in milliseconds
}

function predict() {
  console.log('Predicting');
  $.getJSON('/predict', {
    a: $('input[id="search"]').val(),
  }, function(data) {
    $("#resultsHook").empty()
    if ($('input[id="search"]').val()) {
      for (var i = 0; i < data.result.length; i++) {
        console.log(data.result[i]);
        item = `<div class="resultBox">
                    <p class="result">` + data.result[i] + `</p>
                </div>`
        $jitem = $(item)
        $("#resultsHook").append($jitem);
      }
      attachEvents()
    }

  });
  return false;
}

function attachEvents() {
  $(".result").click(function(event) {
    console.log('test');
    var text = $(event.target).text()
    $('input[id="search"]').val(text)
    predict()
  });
}

jQuery(document).ready(function() {
  $('input[id="search"]').keyup(function(e) {
    console.log('Reseting Timer');
    resetTimer()
  })

  $('.search-button').click(function() {
    $(this).parent().toggleClass('open');
  });
});