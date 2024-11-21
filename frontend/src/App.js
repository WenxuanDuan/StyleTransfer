import React, { useState, useEffect } from "react";
import axios from "axios";

const App = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [style, setStyle] = useState(""); // 默认不选中任何样式
    const [styles, setStyles] = useState([]); // 样式列表
    const [outputImage, setOutputImage] = useState(null);
    const [uploadedImage, setUploadedImage] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    // 加载样式列表
    useEffect(() => {
        const fetchStyles = async () => {
            try {
                const response = await axios.get("http://localhost:5001/styles/list");
                setStyles(response.data); // 更新样式列表
                if (response.data.length > 0) {
                    setStyle(response.data[0]); // 默认选择第一个样式
                }
            } catch (error) {
                console.error("Error fetching styles:", error);
                alert("Failed to load styles. Please try again.");
            }
        };

        fetchStyles();
    }, []); // 仅在组件挂载时执行

    // 处理用户上传文件
    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);
        setUploadedImage(URL.createObjectURL(file));
        setOutputImage(null); // 清空输出图片
    };


    // 处理用户选择风格
    const handleStyleChange = (event) => {
        setStyle(event.target.value);
        setOutputImage(null); // 清空输出图片
    };


    // 处理表单提交
    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!selectedFile || !style) {
            alert("Please upload an image and select a style!");
            return;
        }

        const formData = new FormData();
        formData.append("image", selectedFile);
        formData.append("style", style);

        try {
            setIsLoading(true);
            const response = await axios.post("http://localhost:5001/upload", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });

            const imagePath = response.data.imagePath;
            console.log("Output Image Path:", imagePath);
            setOutputImage(`http://localhost:5001${imagePath}`);
        } catch (error) {
            console.error("Error uploading file:", error);
            alert("Error applying style transfer. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div style={{ padding: "20px" }}>
            <h1>Style Transfer Application</h1>
            <form onSubmit={handleSubmit}>
                <div style={{ marginBottom: "10px" }}>
                    <label htmlFor="file">Upload your image:</label>
                    <input
                        type="file"
                        id="file"
                        onChange={handleFileChange}
                        style={{ marginLeft: "10px" }}
                    />
                </div>
                <div style={{ marginBottom: "10px" }}>
                    <label htmlFor="style">Choose a style:</label>
                    <select
                        id="style"
                        value={style}
                        onChange={handleStyleChange}
                        style={{ marginLeft: "10px" }}
                    >
                        {styles.map((styleName) => (
                            <option key={styleName} value={styleName}>
                                {styleName}
                            </option>
                        ))}
                    </select>
                </div>
                <button type="submit" disabled={isLoading}>
                    Apply Style
                </button>
            </form>

            {isLoading && (
                <div style={{ marginTop: "20px", textAlign: "center", color: "blue" }}>
                    <p>Applying style... Please wait!</p>
                </div>
            )}

            <div style={{ display: "flex", flexWrap: "wrap", marginTop: "20px" }}>
                {uploadedImage && (
                    <div style={{ margin: "10px", textAlign: "center" }}>
                        <h3>Uploaded Image:</h3>
                        <img
                            src={uploadedImage}
                            alt="Uploaded"
                            style={{
                                width: "200px",
                                height: "auto",
                                border: "1px solid #ccc",
                                borderRadius: "10px",
                            }}
                        />
                    </div>
                )}
                {style && (
                    <div style={{ margin: "10px", textAlign: "center" }}>
                        <h3>Style Image:</h3>
                        <img
                            src={`http://localhost:5001/styles/${style}.jpg`}
                            alt="Style"
                            style={{
                                width: "200px",
                                height: "auto",
                                border: "1px solid #ccc",
                                borderRadius: "10px",
                            }}
                        />
                    </div>
                )}
                {outputImage && (
                    <div style={{ margin: "10px", textAlign: "center" }}>
                        <h3>Output Image:</h3>
                        <img
                            src={`${outputImage}?timestamp=${Date.now()}`} // 防止缓存
                            alt="Output"
                            style={{
                                width: "200px",
                                height: "auto",
                                border: "1px solid #ccc",
                                borderRadius: "10px",
                            }}
                        />
                    </div>
                )}
            </div>
        </div>
    );
};

export default App;
