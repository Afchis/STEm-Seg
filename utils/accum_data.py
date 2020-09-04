import torch


class AccumData():
    def init(self):
        self.disp = {
            "train_Sloss" : 0,
            "train_Closs" : 0,
            "train_Eloss" : 0,
            "train_Tloss" : 0,
            "valid_Sloss" : 0,
            "valid_Closs" : 0,
            "valid_Eloss" : 0,
            "valid_Tloss" : 0,
            "train_metric" : 0,
            "valid_metric" : 0,
            "train_cluster_metric" : 0,
            "valid_cluster_metric" : 0,
            "train_iter" : 0,
            "valid_iter" : 0
        }        

    def update(self, key, x):
        if key is "iter_metric":
            self.iter_metric = x
        elif key is "train_iter":
            self.disp[key] += 1
        elif key is "valid_iter":
            self.disp[key] += 1
        else:
            self.disp[key] += x

    def printer_train(self, i):
        if i % 5 == 0:
            print("Train iter:", i, "Loss: %0.4f" % (self.disp["train_Tloss"]/self.disp["train_iter"]),
                  "MetricIoU: %0.4f" % (self.disp["train_metric"]/self.disp["train_iter"]))
            # print(" "*10, "Smooth: %0.4f" % (self.disp["train_Sloss"]/self.disp["train_iter"]),
            #       "Center: %0.4f" % (self.disp["train_Closs"]/self.disp["train_iter"]),
            #       "Ebmedding: %0.4f" % (self.disp["train_Eloss"]/self.disp["train_iter"]))

    def printer_valid(self, i):
        if i % 5 == 0:
            print("Valid iter:", i, "Loss: %0.4f" % (self.disp["valid_Tloss"]/self.disp["valid_iter"]),
                  "MetricIoU: %0.4f" % (self.disp["valid_metric"]/self.disp["valid_iter"]))
            # print(" "*10, "Smooth: %0.4f" % (self.disp["valid_Sloss"]/self.disp["valid_iter"]),
            #       "Center: %0.4f" % (self.disp["valid_Closs"]/self.disp["valid_iter"]),
            #       "Ebmedding: %0.4f" % (self.disp["valid_Eloss"]/self.disp["valid_iter"]))

    def printer_epoch(self):
        print(" "*15, "Train --> Loss: %0.3f" % (self.disp["train_Tloss"]/self.disp["train_iter"]),
              "IoU: %0.3f" % (self.disp["train_metric"]/self.disp["train_iter"]))
        print(" "*15, "Valid --> Loss: %0.3f" % (self.disp["valid_Tloss"]/self.disp["valid_iter"]),
              "IoU: %0.3f" % (self.disp["valid_metric"]/self.disp["valid_iter"]))

    def tensorboard(self, writer, tb, epoch):
        train_Tloss = self.disp["train_Tloss"]/self.disp["train_iter"]
        train_Sloss = self.disp["train_Sloss"]/self.disp["train_iter"]
        train_Closs = self.disp["train_Closs"]/self.disp["train_iter"]
        train_Eloss = self.disp["train_Eloss"]/self.disp["train_iter"]
        train_metric = self.disp["train_metric"]/self.disp["train_iter"]
        valid_Tloss = self.disp["valid_Tloss"]/self.disp["valid_iter"]
        valid_Sloss = self.disp["valid_Sloss"]/self.disp["valid_iter"]
        valid_Closs = self.disp["valid_Closs"]/self.disp["valid_iter"]
        valid_Eloss = self.disp["valid_Eloss"]/self.disp["valid_iter"]
        valid_metric = self.disp["valid_metric"]/self.disp["valid_iter"]
        if tb is not "None":
            writer.add_scalars('%s_Total_loss' % tb, {'train' : train_Tloss,
                                                      'valid' : valid_Tloss}, epoch)
            writer.add_scalars('%s_Smooth_loss' % tb, {'train' : train_Sloss,
                                                       'valid' : valid_Sloss}, epoch)
            writer.add_scalars('%s_Center_loss' % tb, {'train' : train_Closs,
                                                       'valid' : valid_Closs}, epoch)
            writer.add_scalars('%s_Embeddidg_loss' % tb, {'train' : train_Eloss,
                                                          'valid' : valid_Eloss}, epoch)
            writer.add_scalars('%s_IoU_metric' % tb, {'train' : train_metric,
                                                      'valid' : valid_metric}, epoch)

    def visual_train(self, vis, Visual, pred_masks, outs, i, mode):
        if vis is True:
            Heat_map, _, _ = outs
            Visual(pred_masks, Heat_map, i, mode)

    def visual_inference(self, vis, Visual_inference, pred_clusters, images, i, mode):
        if vis is True:
            Visual_inference(pred_clusters, images, i, mode)

