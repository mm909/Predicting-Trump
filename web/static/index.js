t = setTimeout(predict, 500);
clearTimeout(t);

function resetTimer() {
  clearTimeout(t);
  t = setTimeout(predict, 500); // time is in milliseconds
}

Start = new Date();
End = new Date();
StartTime = Start.getTime();
EndTime = End.getTime();

function predict() {
  Start = new Date();
  StartTime = Start.getTime();
  let timeInterval = window.setInterval(function() {
    End = new Date();
    EndTime = End.getTime();
    $('.Timer').text((EndTime - StartTime) / 1000 + " Seconds")
  }, 10);
  // console.log('Predicting');
  if ($('input[id="search"]').val() != '') {
    $.getJSON('/predict', {
      a: $('input[id="search"]').val(),
    }, function(data) {
      $("#resultsHook").empty()
      for (var i = 0; i < data.result.length; i++) {
        item = `<div class="resultBox">
                    <p class="result">` + data.result[i] + `</p>
                </div>`
        $jitem = $(item)
        $("#resultsHook").append($jitem);
      }
      attachEvents()
      End = new Date();
      EndTime = End.getTime();
      // console.log((EndTime - StartTime) / 1000);
      clearInterval(timeInterval)
      $('.Timer').text((EndTime - StartTime) / 1000 + " Seconds")
      $('.Sentences').text(data.result.length + " Sentences")

      chars = 0;
      for (var i = 0; i < data.result.length; i++) {
        chars += data.result[i].length - $('input[id="search"]').val().length
      }
      $('.Letters').text(chars + " Letters")



    });
  } else {
    $("#resultsHook").empty()
    clearInterval(timeInterval)
    clearStats()
  }

  return false;
}

function clearStats() {
  $('.Timer').text("0.00 Seconds")
  $('.Sentences').text("0 Sentences")
  $('.Letters').text("0 Letters")
}

function attachEvents() {
  $(".result").click(function(event) {
    // console.log('test');
    var text = $(event.target).text()
    $('input[id="search"]').val(text)
    predict()
  });
}

jQuery(document).ready(function() {
  $('input[id="search"]').keyup(function(e) {
    // console.log('Reseting Timer');
    resetTimer()
  })
  $("#resultsHook").css("display", "none");

  $('.search-button').click(function() {
    $(this).parent().toggleClass('open');
    if ($('#resultsHook').css('display') == 'none') {
      // console.log("test");
      $("#resultsHook").css("display", "block");
    } else {
      clearStats()
      $("#resultsHook").css("display", "none");
      $("#resultsHook").empty()
      $('input[id="search"]').val('')
    }
  });
});