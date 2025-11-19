mergeInto(LibraryManager.library, {
    // Stores blob parts for chunked downloads
    _blobParts: {},

    // Start a new chunked download
    DownloadFileBegin: function(fileNamePtr) {
        var fileName = UTF8ToString(fileNamePtr);

        // Create a new array to store blob parts for this file
        if (!this._blobParts) {
            this._blobParts = {};
        }
        this._blobParts[fileName] = [];
    },

    // Add a chunk to the download
    DownloadFileAddChunk: function(fileNamePtr, dataPtr, dataLength) {
        var fileName = UTF8ToString(fileNamePtr);

        if (!this._blobParts || !this._blobParts[fileName]) {
            console.error("DownloadFileBegin must be called first");
            return false;
        }

        try {
            // Convert to unsigned 32-bit integer to handle negative values
            var offset = dataPtr >>> 0;

            // Validate offset
            if (offset < 0 || offset + dataLength > HEAPU8.buffer.byteLength) {
                console.error("Invalid pointer:", {
                    dataPtr: dataPtr,
                    offset: offset,
                    dataLength: dataLength,
                    bufferLength: HEAPU8.buffer.byteLength
                });
                return false;
            }

            // Create view and copy the data
            var data = new Uint8Array(HEAPU8.buffer, offset, dataLength);
            var dataCopy = new Uint8Array(data);

            // Add to blob parts
            this._blobParts[fileName].push(dataCopy);

            return true;
        } catch (e) {
            console.error("Error in DownloadFileAddChunk:", e, {
                dataPtr: dataPtr,
                dataLength: dataLength
            });
            return false;
        }
    },

    // Finalize and trigger the download
    DownloadFileEnd: function(fileNamePtr) {
        var fileName = UTF8ToString(fileNamePtr);

        if (!this._blobParts || !this._blobParts[fileName]) {
            console.error("No blob parts found for file: " + fileName);
            return false;
        }

        // Create blob from all parts
        var blob = new Blob(this._blobParts[fileName], { type: 'application/octet-stream' });

        // Clean up blob parts
        delete this._blobParts[fileName];

        // Create download
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = fileName;

        document.body.appendChild(a);
        a.click();

        // Cleanup
        setTimeout(function() {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 100);

        return true;
    }
});
