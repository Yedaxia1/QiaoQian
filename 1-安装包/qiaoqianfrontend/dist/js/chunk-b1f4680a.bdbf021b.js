(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-b1f4680a"],{1816:function(e,t,i){e.exports=i.p+"img/ocr-active.1ad5e422.svg"},"3a78":function(e,t,i){e.exports=i.p+"img/txt-active.2a833355.svg"},"5ecd":function(e,t,i){"use strict";var n=i("9905"),s=i.n(n);s.a},"61cf":function(e,t,i){"use strict";var n=function(){var e=this,t=e.$createElement,i=e._self._c||t;return i("div",{staticClass:"process-item",on:{click:e.itemClick}},[e.isActive?i("div",{staticClass:"item-icon"},[e._t("icon-active")],2):i("div",{staticClass:"item-icon"},[e._t("icon")],2),i("div",{staticClass:"item-text"},[e._t("text")],2)])},s=[],c={name:"ProcessItem",data:function(){return{}},props:{itemName:String,isActive:Boolean},methods:{itemClick:function(){var e=this.itemName;this.$emit("item-click",e)}}},a=c,o=(i("5ecd"),i("2877")),r=Object(o["a"])(a,n,s,!1,null,"3206c9ce",null);t["a"]=r.exports},"92f0b":function(e,t,i){e.exports=i.p+"img/saomiaojian.1070c07f.svg"},9905:function(e,t,i){},"9bb4":function(e,t,i){e.exports=i.p+"img/saomiaojian-active.647d24a8.svg"},a83a:function(e,t,i){e.exports=i.p+"img/txt.6b43539d.svg"},b4c1:function(e,t,i){},c410:function(e,t,i){"use strict";var n=i("b4c1"),s=i.n(n);s.a},c665:function(e,t,i){e.exports=i.p+"img/ocr.3e256736.svg"},d967:function(e,t,i){"use strict";i.r(t);var n=function(){var e=this,t=e.$createElement,i=e._self._c||t;return i("div",{staticClass:"img-proc"},[i("el-header",[i("nav-bar",{scopedSlots:e._u([{key:"left",fn:function(){return[i("i",{staticClass:"el-icon-back",on:{click:e.back}})]},proxy:!0},{key:"center",fn:function(){return[i("span",[e._v(e._s(e.currentModule))])]},proxy:!0},{key:"right",fn:function(){return[i("i",{staticClass:"el-icon-check",on:{click:e.finish}})]},proxy:!0}])})],1),i("el-dialog",{attrs:{title:"识别结果",visible:e.dialogVisiable,width:"75%"},on:{"update:visible":function(t){e.dialogVisiable=t}}},e._l(e.ocrResult.words_result_num,(function(t){return i("p",{key:t},[e._v(e._s(e.ocrResult.words_result[t-1].words))])})),0),i("el-main",{directives:[{name:"loading",rawName:"v-loading",value:e.fullScreenLoading,expression:"fullScreenLoading"}],attrs:{"element-loading-text":"处理中...","element-loading-background":"rgba(0, 0, 0, 0.6)"}},[i("div",{staticClass:"img-container"},[i("el-image",{attrs:{src:e.userImg,fit:"scale-down"}})],1)]),i("el-footer",{staticClass:"tabs"},[i("el-tabs",[e._l(e.modules,(function(t){return[i("el-tab-pane",{scopedSlots:e._u([{key:"label",fn:function(){return[i("process-item",{attrs:{itemName:t.name,isActive:t.isActive},on:{"item-click":e.handleClick},scopedSlots:e._u([{key:"icon",fn:function(){return[i("img",{attrs:{src:t.icon}})]},proxy:!0},{key:"icon-active",fn:function(){return[i("img",{attrs:{src:t.iconActive}})]},proxy:!0},{key:"text",fn:function(){return[i("span",[e._v(e._s(t.name))])]},proxy:!0}],null,!0)})]},proxy:!0}],null,!0)})]}))],2)],1)],1)},s=[],c=(i("c740"),i("6259")),a=i("61cf"),o=i("c665"),r=i.n(o),l=i("1816"),u=i.n(l),d=i("92f0b"),m=i.n(d),f=i("9bb4"),g=i.n(f),v=i("a83a"),p=i.n(v),h=i("3a78"),b=i.n(h),k=(i("de33"),i("bc3a")),x=i.n(k),_=i("025e");x.a.defaults.baseURL=Object(_["a"])();var w={name:"FileProc",components:{NavBar:c["a"],ProcessItem:a["a"]},data:function(){return{modules:[{name:"OCR",icon:r.a,iconActive:u.a,isActive:!1},{name:"扫描件",icon:m.a,iconActive:g.a,isActive:!1},{name:"保存txt",icon:p.a,iconActive:b.a,isActive:!1}],currentModule:"文档处理",fullScreenLoading:!1,dialogVisiable:!1,ocrResult:"",previousIndex:0}},computed:{userImg:function(){return this.$store.state.editedImg}},activated:function(){this.init()},methods:{init:function(){this.currentModule="文档处理"},back:function(){this.$router.back()},finish:function(){var e=new Date,t=document.createElement("a");t.href=this.$store.state.editedImg,t.setAttribute("download","扫描结果-"+e.getFullYear()+"-"+(e.getMonth()+1)+"-"+e.getDate()+"-"+e.getHours()+e.getMinutes()+e.getSeconds()+".jpg"),t.click()},handleClick:function(e){var t=this,i=this.$store.state.findIndex(this.modules,e,"name");this.modules[i].isActive=!0,this.modules[this.previousIndex].isActive=!1,this.previousIndex=i,this.currentModule=e;var n=new FormData;switch(e){case"OCR":this.fullScreenLoading=!0,n.append("editedImg",this.$store.state.editedImg),x.a.post("imgProc/ocr-proc/",n).then((function(e){var i=e.data.error_code;1008===i?(t.fullScreenLoading=!1,t.$message.error("识别失败，请稍后再试")):0===i&&(t.ocrResult=e.data.ocr_result,t.fullScreenLoading=!1,t.$message.success("识别成功!"),t.dialogVisiable=!0)})).catch((function(e){console.log(e),t.fullScreenLoading=!1,t.$message.error("识别失败，请检查您的网络连接")}));break;case"扫描件":this.fullScreenLoading=!0,n.append("editedImg",this.$store.state.editedImg),x.a.post("imgProc/picScanner/",n).then((function(e){var i=e.data.error_code;0===i?(t.$store.commit("saveImg",e.data.img_proc_result),t.$message.success("处理成功！")):1008===i&&t.$message.error("处理失败，请稍后再试"),t.fullScreenLoading=!1})).catch((function(e){console.log(e),t.$message.error("处理失败，请检查您的网络连接"),t.fullScreenLoading=!1}));break;case"保存txt":for(var s=new Date,c="",a=0;a<this.ocrResult.words_result_num;a++)c+=this.ocrResult.words_result[a].words+"\n";var o=document.createElement("a");o.setAttribute("href","data:text/plain;charset=utf-8,"+c),o.setAttribute("download","识别结果-"+s.getFullYear()+"-"+(s.getMonth()+1)+"-"+s.getDate()+"-"+s.getHours()+s.getMinutes()+s.getSeconds()),o.click();break;default:break}}}},S=w,A=(i("c410"),i("2877")),$=Object(A["a"])(S,n,s,!1,null,"6f40453f",null);t["default"]=$.exports},de33:function(e,t,i){e.exports=i.p+"img/theme2.6e63ca05.jpg"}}]);
//# sourceMappingURL=chunk-b1f4680a.bdbf021b.js.map