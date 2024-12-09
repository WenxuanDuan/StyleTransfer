/*
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

*/

import React, { useState, useEffect } from "react";
import { LazyLoadImage } from "react-lazy-load-image-component";
import "./App.css";

const App = () => {
    const [styles, setStyles] = useState([]);
    const [selectedStyle, setSelectedStyle] = useState("");
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadedImage, setUploadedImage] = useState(null);
    const [outputImage, setOutputImage] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isHovered, setIsHovered] = useState(false); // 控制蒙版显示状态

    useEffect(() => {
        const fetchStyles = async () => {
            try {
                const response = await fetch("http://localhost:5001/styles/list");
                const data = await response.json();
                setStyles(data);
                if (data.length > 0) setSelectedStyle(data[0].styleName);
            } catch (error) {
                console.error("Error fetching styles:", error);
            }
        };
        fetchStyles();
    }, []);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);
        setUploadedImage(URL.createObjectURL(file)); // 生成文件的临时 URL
        setOutputImage(null); // 重置输出图片
    };

    const handleStyleSelect = (styleName) => {
        setSelectedStyle(styleName);
    };

    const handleApplyStyle = async () => {
        if (!selectedFile) {
            alert("Please upload an image!");
            return;
        }
        if (!selectedStyle) {
            alert("Please select a style!");
            return;
        }

        const formData = new FormData();
        formData.append("image", selectedFile);
        formData.append("style", selectedStyle);

        try {
            setIsLoading(true); // 开始加载
            const response = await fetch("http://localhost:5001/upload", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Failed to apply style: ${response.statusText}`);
            }

            const data = await response.json();
            console.log("Response data:", data);

            if (data.imagePath) {
                setOutputImage(`http://localhost:5001${data.imagePath}`); // 设置输出图片
            } else {
                throw new Error("Invalid response from server.");
            }
        } catch (error) {
            console.error("Error applying style:", error);
            alert("Failed to apply style. Please try again.");
        } finally {
            setIsLoading(false); // 停止加载
        }
    };

    const handleDownload = async () => {
        if (!outputImage) {
            alert("No styled image available to download.");
            return;
        }

        try {
            const response = await fetch(outputImage);
            if (!response.ok) {
                throw new Error(`Failed to fetch image: ${response.statusText}`);
            }
            const blob = await response.blob(); // 将响应转为 Blob 对象
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob); // 创建临时 URL
            link.download = "styled_image.jpg"; // 设置下载文件名
            document.body.appendChild(link);
            link.click(); // 触发下载
            document.body.removeChild(link); // 移除临时链接
        } catch (error) {
            console.error("Error downloading image:", error);
            alert("Failed to download the styled image.");
        }
    };


    return (
        <div className="App">
            <div className="app-container">
                <h1 className="app-title">
                    🎨 <span>Style Transfer Application</span>
                </h1>

                <div className="upload-section">
                    <h5>Upload Your Image</h5>
                    <input type="file" onChange={handleFileChange} />
                </div>

                <div className="image-display">
                    {/* 上传图片 */}
                    {uploadedImage && (
                        <div className="image-container">
                            <h6>Uploaded Image:</h6>
                            <img src={uploadedImage} alt="Uploaded Preview" />
                        </div>
                    )}

                    {/* 输出图片 */}
                    {outputImage && (
                        <div
                            className="image-container"
                            onMouseEnter={() => setIsHovered(true)} // 显示蒙版
                            onMouseLeave={() => setIsHovered(false)} // 隐藏蒙版
                        >
                            <h6>Styled Image:</h6>
                            <div className="output-image-wrapper">
                                <img src={outputImage} alt="Styled Output" />
                                {isHovered && (
                                    <div
                                        className="download-overlay"
                                        onClick={handleDownload}
                                    >
                                        <i className="fas fa-download"></i> Download
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                <div className="style-section">
                    <h6>Select a Style</h6>
                    <div className="style-grid">
                        {styles.map(({ styleName, imagePath }) => (
                            <div
                                key={styleName}
                                className={`style-card ${
                                    selectedStyle === styleName ? "selected" : ""
                                }`}
                                onClick={() => handleStyleSelect(styleName)}
                            >
                                <LazyLoadImage
                                    src={`http://localhost:5001${imagePath}`}
                                    alt={styleName}
                                    effect="blur"
                                    onError={(e) =>
                                        console.error(`Error loading thumbnail for ${styleName}:`, e)
                                    }
                                />
                                <p>{styleName}</p>
                            </div>
                        ))}
                    </div>
                </div>

                <button
                    className="apply-btn"
                    onClick={handleApplyStyle}
                    disabled={isLoading}
                >
                    {isLoading ? "Applying Style..." : "Apply Style"}
                </button>
            </div>
        </div>
    );
};

export default App;


