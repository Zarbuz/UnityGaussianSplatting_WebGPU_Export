mergeInto(LibraryManager.library, {
    PickFile: function(acceptedExtensions, gameObjectNamePtr, callbackMethodNamePtr) {
        var gameObjectName = UTF8ToString(gameObjectNamePtr);
        var callbackMethodName = UTF8ToString(callbackMethodNamePtr);
        var accept = UTF8ToString(acceptedExtensions);

        // Create a file input element
        var fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = accept;
        fileInput.style.display = 'none';

        fileInput.onchange = function(event) {
            var file = event.target.files[0];
            if (file) {
                var reader = new FileReader();

                reader.onload = function(e) {
                    var arrayBuffer = e.target.result;
                    var uint8Array = new Uint8Array(arrayBuffer);

                    // Allocate memory in Unity's heap for the file data
                    var bufferSize = uint8Array.length;
                    var buffer = _malloc(bufferSize);

                    // Copy data to Unity's heap
                    HEAPU8.set(uint8Array, buffer);

                    // Create message with filename and buffer info
                    // Format: "filename|bufferPtr|bufferSize"
                    var message = file.name + '|' + buffer + '|' + bufferSize;

                    // Send message to Unity
                    SendMessage(gameObjectName, callbackMethodName, message);
                };

                reader.readAsArrayBuffer(file);
            }

            // Remove the input element
            document.body.removeChild(fileInput);
        };

        fileInput.oncancel = function() {
            document.body.removeChild(fileInput);
        };

        // Append to body and click
        document.body.appendChild(fileInput);
        fileInput.click();
    },

    FreeFileBuffer: function(bufferPtr) {
        _free(bufferPtr);
    }
});
